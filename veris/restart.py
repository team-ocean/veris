"""Complete version 1 serial checkpoints, distinct from visualization output.

Every array-only State field retains native precision and full boundary halos.
Definition inputs use typed numeric groups; exact JSON integer clocks preserve
absolute forcing position. Cold validation precedes backend enumeration and
placement. Restored Diagnostics are None until another physical timestep.
Unfinished/crossing means are discarded; output resumes on its original cadence
in a new file. Distributed/multi-device restart is explicitly unsupported.
"""

import json
import os
import tempfile
from dataclasses import asdict, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import numpy as np

from veris._typing import State
from veris.experiment import Experiment
from veris.experiment_file import (
    _nonfinite_constant,
    _unique_object,
    experiment_from_dict,
    experiment_to_dict,
)
from veris.io.calendar import Calendar, duration_us
from veris.io.configuration import OutputSettings
from veris.io.guard import require_host
from veris.io.schedule import iter_segments
from veris.io.storage import NetCDFWriter, _json_scalar, selected_fields
from veris.model_initialization import _validate_inputs
from veris.restart_arrays import ArrayManifest, hydrate_definition, validate_links
from veris.simulation_io import _new_path
from veris.variables import VARIABLES

_NAMES = tuple(item.name for item in fields(State))
_RECORD_NAMES = {"time", "time_bounds", "sample_count", "partial"}
_ATTRIBUTES = {
    "Conventions",
    "veris_file_kind",
    "veris_restart_version",
    "backend",
    "clock",
    "experiment",
    "configuration",
    "physical_constants",
}


def _json(value: Any) -> str:
    """Encode small metadata with no nonfinite extensions or numeric coercion."""
    return json.dumps(value, default=_json_scalar, allow_nan=False)


def _decode(value: Any, context: str) -> dict[str, Any]:
    """Strictly decode one metadata object, including all nested JSON objects."""
    try:
        result = json.loads(
            value, object_pairs_hook=_unique_object, parse_constant=_nonfinite_constant
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid {context} JSON metadata: {error}") from error
    if not isinstance(result, dict):
        raise TypeError(f"{context} metadata must be an object")
    return result


def _serial(experiment: Experiment, backend: str) -> None:
    """Validate requested serial execution and its recorded actual backend."""
    if (
        experiment.execution.partitions != (1, 1)
        or experiment.configuration.use_sharding
    ):
        raise ValueError(
            "complete restart supports serial execution only; distributed partitions unsupported"
        )
    if backend not in ("cpu", "gpu"):
        raise ValueError("restart backend must be cpu or gpu")
    if experiment.execution.backend not in (None, backend):
        raise ValueError("saved backend disagrees with Execution backend")


def _clock(experiment: Experiment, clock: dict[str, Any]) -> tuple[int, timedelta]:
    """Validate exact absolute clocks without float or signed-int64 conversion."""
    if set(clock) != {"completed_steps", "elapsed_microseconds", "forcing_step"}:
        raise ValueError("restart clock requires complete known keys")
    if any(type(value) is not int or value < 0 for value in clock.values()):
        raise ValueError("restart clock counts must be nonnegative integers")
    completed = clock["completed_steps"]
    if clock["forcing_step"] != completed:
        raise ValueError("restart forcing cursor must equal completed_steps")
    if clock["elapsed_microseconds"] != completed * duration_us(
        experiment.step_duration
    ):
        raise ValueError("restart elapsed clock disagrees with completed steps")
    length = experiment.forcing.length
    if length is not None and completed > length:
        raise ValueError("restart forcing cursor exceeds series length")
    try:
        elapsed = timedelta(microseconds=clock["elapsed_microseconds"])
    except OverflowError as error:
        raise ValueError("restart elapsed clock exceeds timedelta range") from error
    output = experiment.output
    if output is not None and output.enabled:
        Calendar(output.start, output.calendar).date_at(elapsed)
    return completed, elapsed


def _calendar(experiment: Experiment) -> Calendar:
    """Provide CF tooling coordinates without imposing a date on a model clock."""
    settings = experiment.output if experiment.output is not None else OutputSettings()
    return Calendar(settings.start, settings.calendar)


def _state_arrays(source: State, experiment: Experiment) -> tuple[dict[str, Any], str]:
    """Preflight a live single-device State before creating any file resources."""
    devices: set[Any] = set()
    for name in _NAMES:
        value = getattr(source, name)
        if not isinstance(value, jax.Array) or not value.is_fully_addressable:
            raise ValueError(
                f"restart field {name} requires addressable serial JAX State"
            )
        if value.sharding.num_devices != 1:
            raise ValueError("complete restart rejects multi-device/distributed State")
        devices.update(value.devices())
    if len(devices) != 1:
        raise ValueError("complete restart requires all State fields on one device")
    platform = next(iter(devices)).platform
    backend = "gpu" if platform in ("gpu", "cuda", "rocm") else platform
    _serial(experiment, backend)
    arrays = selected_fields(source, _NAMES)
    shape = experiment.grid.storage_shape()
    dtype = np.dtype(experiment.configuration.dtype)
    for name, value in arrays.items():
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"restart field {name} has incompatible shape/dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"restart field {name} must be finite")
    return arrays, backend


def save_restart(
    path: str | Path,
    state: State,
    experiment: Experiment,
    completed_steps: int,
    elapsed: timedelta,
) -> None:
    """Atomically publish a new complete serial checkpoint without overwriting."""
    require_host()
    destination = _new_path(path)
    arrays, backend = _state_arrays(state, experiment)
    _validate_inputs(experiment)
    clock = {
        "completed_steps": completed_steps,
        "elapsed_microseconds": duration_us(elapsed),
        "forcing_step": completed_steps,
    }
    _clock(experiment, clock)
    manifest = ArrayManifest()
    definition = experiment_to_dict(experiment, _array_encoder=manifest.encode)
    attributes = {
        "veris_file_kind": "restart",
        "backend": backend,
        "clock": _json(clock),
        "experiment": _json(definition),
        "configuration": _json(asdict(experiment.configuration)),
        "physical_constants": _json(asdict(experiment.physical_constants)),
    }
    calendar = _calendar(experiment)
    time = duration_us(elapsed) / 1_000_000
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        with NetCDFWriter(
            temporary,
            calendar=calendar.name,
            units=calendar.units,
            attributes=attributes,
        ) as writer:
            writer.append("state", arrays, time=time, bounds=(time, time))
        with h5netcdf.File(temporary, "r+") as file:
            file.attrs["veris_restart_version"] = 1
            manifest.write(file)
        # link publishes without clobbering even if another writer wins a race.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _record(
    file: h5netcdf.File, experiment: Experiment, elapsed: timedelta
) -> dict[str, Any]:
    """Validate the exact single instantaneous full-halo State record schema."""
    if set(file.groups) != {"state", "definition"} or file.variables or file.dimensions:
        raise ValueError("unexpected restart root groups/variables/dimensions")
    group = file.groups["state"]
    expected = set(_NAMES) | _RECORD_NAMES
    actual = set(group.variables)
    if actual != expected:
        raise ValueError(
            f"incomplete or unexpected restart State fields: missing {sorted(expected - actual)}, unknown {sorted(actual - expected)}"
        )
    if group.groups or dict(group.attrs) != {"mean": 0}:
        raise ValueError("restart State requires one instantaneous record, not a mean")
    shape = experiment.grid.storage_shape()
    dimensions = {"time": 1, "bounds": 2}
    for name in _NAMES:
        dimensions.update(dict(zip(VARIABLES[name].dimensions, shape, strict=True)))
    if {name: len(value) for name, value in group.dimensions.items()} != dimensions:
        raise ValueError("restart State has incompatible grid dimensions/shape")
    if not group.dimensions["time"].isunlimited() or any(
        value.isunlimited()
        for name, value in group.dimensions.items()
        if name != "time"
    ):
        raise ValueError("restart State has incompatible unlimited dimensions")
    specs = {
        "time": (("time",), "float64"),
        "time_bounds": (("time", "bounds"), "float64"),
        "sample_count": (("time",), "int64"),
        "partial": (("time",), "int8"),
    }
    calendar = _calendar(experiment)
    time_attrs = {
        "units": calendar.units,
        "calendar": calendar.name,
        "bounds": "time_bounds",
        "standard_name": "time",
    }
    for name, (dims, dtype) in specs.items():
        variable = group.variables[name]
        if (
            variable.dimensions != dims
            or variable.dtype != np.dtype(dtype)
            or variable.shape != tuple(dimensions[name] for name in dims)
        ):
            raise ValueError(f"restart record {name} has incompatible dimensions/dtype")
        if dict(variable.attrs) != (time_attrs if name == "time" else {}):
            raise ValueError(f"restart record {name} has incompatible metadata")
    time = duration_us(elapsed) / 1_000_000
    if float(group.variables["time"][0]) != time or not np.array_equal(
        group.variables["time_bounds"][0], (time, time)
    ):
        raise ValueError("restart CF time disagrees with exact elapsed clock")
    if (
        int(group.variables["partial"][0]) != 0
        or int(group.variables["sample_count"][0]) != 1
    ):
        raise ValueError("restart requires a complete nonpartial single sample")
    arrays = {}
    for name in _NAMES:
        variable = group.variables[name]
        if variable.dimensions != (
            "time",
            *VARIABLES[name].dimensions,
        ) or variable.shape != (1, *shape):
            raise ValueError(f"restart field {name} has incompatible dimensions/shape")
        if variable.dtype != np.dtype(experiment.configuration.dtype):
            raise ValueError(f"restart field {name} has incompatible dtype/precision")
        expected_attrs = {
            **VARIABLES[name].netcdf_attributes(),
            "cell_methods": "time: point",
        }
        if dict(variable.attrs) != expected_attrs:
            raise ValueError(f"restart field {name} has incompatible registry metadata")
        value = np.asarray(variable[0])
        if not np.isfinite(value).all():
            raise ValueError(f"restart field {name} must be finite")
        arrays[name] = value
    return arrays


def _output(
    experiment: Experiment,
    path: str | Path | None,
    settings: OutputSettings | None,
    completed: int,
    elapsed: timedelta,
) -> Experiment:
    """Validate a fresh output destination and unchanged calendar/origin."""
    original = experiment.output
    if settings is not None:
        if not isinstance(settings, OutputSettings):
            raise TypeError("output_settings must be OutputSettings or None")
        saved = original if original is not None else OutputSettings()
        if Calendar(saved.start, saved.calendar) != Calendar(
            settings.start, settings.calendar
        ):
            raise ValueError(
                "restart output override cannot change calendar or origin/start"
            )
        experiment = replace(experiment, output=settings)
    output = experiment.output
    if output is not None and output.enabled:
        if path is None:
            raise ValueError("enabled restart output requires a new output_path")
        _new_path(path)
        Calendar(output.start, output.calendar).date_at(elapsed)
        iter_segments(
            output,
            experiment.step_duration.total_seconds(),
            max(0, experiment.steps - completed),
            start_step=completed,
        )
    return experiment


def read_restart(
    path: str | Path,
    *,
    output_path: str | Path | None = None,
    output_settings: OutputSettings | None = None,
) -> tuple[Experiment, State, int, timedelta]:
    """Cold-validate a complete checkpoint, then place its exact full-halo State."""
    require_host()
    source = Path(path)
    try:
        validate_links(source)
        with h5netcdf.File(source, "r") as file:
            if file.attrs.get("veris_file_kind") != "restart":
                raise ValueError(
                    "file is not a complete restart; selected snapshots/means unsupported"
                )
            version = file.attrs.get("veris_restart_version")
            if (
                isinstance(version, (bool, np.bool_))
                or not isinstance(version, (int, np.integer))
                or version != 1
            ):
                raise ValueError("unsupported restart schema version; expected 1")
            if set(file.attrs) != _ATTRIBUTES or file.attrs["Conventions"] != "CF-1.10":
                raise ValueError("incomplete or unexpected restart metadata")
            encoded_definition = _decode(file.attrs["experiment"], "experiment")
            execution = encoded_definition.get("execution")
            if not isinstance(execution, dict) or set(execution) != {
                "backend",
                "partitions",
            }:
                raise ValueError("restart requires complete execution metadata")
            partitions = execution["partitions"]
            if (
                not isinstance(partitions, list)
                or partitions != [1, 1]
                or any(type(value) is not int for value in partitions)
            ):
                raise ValueError(
                    "complete restart requires serial partitions; distributed restart unsupported"
                )
            saved_backend = file.attrs["backend"]
            if saved_backend not in ("cpu", "gpu") or execution["backend"] not in (
                None,
                saved_backend,
            ):
                raise ValueError("restart backend disagrees with execution metadata")
            definition = hydrate_definition(file, encoded_definition)
            experiment = experiment_from_dict(definition)
            canonical = experiment_to_dict(
                experiment, _array_encoder=ArrayManifest().encode
            )
            if json.dumps(encoded_definition, sort_keys=True) != json.dumps(
                canonical, sort_keys=True
            ):
                raise ValueError(
                    "incomplete or incompatible resolved experiment metadata"
                )
            backend = str(file.attrs["backend"])
            _serial(experiment, backend)
            _validate_inputs(experiment)
            completed, elapsed = _clock(
                experiment, _decode(file.attrs["clock"], "clock")
            )
            for name, expected in (
                ("configuration", experiment.configuration),
                ("physical_constants", experiment.physical_constants),
            ):
                if _decode(file.attrs[name], name) != json.loads(
                    _json(asdict(expected))
                ):
                    raise ValueError(f"incompatible resolved {name} metadata")
            arrays = _record(file, experiment, elapsed)
        experiment = _output(
            experiment, output_path, output_settings, completed, elapsed
        )
        # Dtype preflight also avoids silently migrating precision on cold reads.
        if jax.dtypes.canonicalize_dtype(experiment.configuration.dtype) != np.dtype(
            experiment.configuration.dtype
        ):
            raise ValueError(
                "restart precision requires jax_enable_x64 before restoration"
            )
    except FileExistsError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
    ) as error:
        raise ValueError(f"{source}: {error}") from error
    # No device enumeration occurs until every byte and host control validates.
    device = jax.devices(backend)[0]
    with jax.default_device(device):
        state = State(
            **{name: jax.device_put(value, device) for name, value in arrays.items()}
        )
        # Persisted cyclic Growth halos are reference-significant. They were
        # prepared by the saved trajectory and must not be rederived/refreshed;
        # the next pure transition owns its normal boundary updates.
        jax.block_until_ready(state)
    return experiment, state, completed, elapsed
