"""Host I/O helpers for the lifecycle, reusing maintained storage and validation.

Histories keep the original calendar and reduction policy. Selected inputs are
serial instantaneous initial/forcing fields, not geometry edits or restarts;
model time, immutable prescribed forcing and last-step Diagnostics stay intact.
All candidate fields validate before State commit, including derived quantities.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from veris.execution.experiment_initialization import _validate_derived_inputs
from veris.execution.rollout_output import output_callbacks
from veris.grid.halos import fill_state_overlap
from veris.io.configuration import OutputSettings
from veris.io.distributed import distributed_collector
from veris.io.output import OutputManager
from veris.io.scan import ScheduledObserver
from veris.io.schedule import iter_segments
from veris.io.storage import (
    ArrayFields,
    _json_scalar,
    read_record,
    storage_fields,
    update_state,
    write_snapshot,
)
from veris.model.configuration import Configuration
from veris.model.experiment import INITIAL_FIELDS, Experiment
from veris.model.forcing import (
    FORCING_FIELDS,
    Forcing,
    _freeze_fields,
    _precision,
    _validate_shapes,
)
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State


def _new_path(path: str | Path) -> Path:
    """Reject overwrites and unreachable destinations before allocation/write."""
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"output parent directory does not exist: {destination.parent}"
        )
    return destination


def validate_output(experiment: Experiment, path: str | Path | None) -> None:
    """Preflight enabled output's path and reachable sampling before allocation."""
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    if experiment.output is None or not experiment.output.enabled:
        return
    if path is None:
        raise ValueError("enabled output requires an output_path")
    _new_path(path)
    # This call validates the complete calendar range eagerly; iteration is lazy.
    iter_segments(
        experiment.output, experiment.step_duration.total_seconds(), experiment.steps
    )


def metadata(
    experiment: Experiment,
    conf: Configuration,
    phys: PhysicalConstants,
    mesh: Mesh | None = None,
) -> dict[str, str]:
    """Encode resolved definition alongside established numerical metadata."""
    from veris.execution.runtime import _execution_metadata

    default = jax.config.jax_default_device
    backend = (
        mesh.devices.flat[0].platform
        if mesh is not None
        else experiment.execution.backend
        or getattr(default, "platform", None)
        or jax.default_backend()
    )
    partitions = (1, 1) if mesh is None else (mesh.shape["x"], mesh.shape["y"])
    return {
        "execution": json.dumps(dict(_execution_metadata(backend, partitions))),
        "configuration": json.dumps(asdict(conf), default=_json_scalar),
        "physical_constants": json.dumps(asdict(phys), default=_json_scalar),
        "experiment": json.dumps(experiment.to_dict()),
    }


def output_resources(
    experiment: Experiment,
    path: str | Path | None,
    conf: Configuration,
    phys: PhysicalConstants,
    mesh: Mesh | None,
    *,
    start_step: int = 0,
    mean_coverage_start_us: int = 0,
    emit_initial_instant: bool = True,
) -> tuple[OutputManager | None, ScheduledObserver | None]:
    """Prepare a lazy writer and persistent reducer only for enabled output."""
    if experiment.output is None or not experiment.output.enabled:
        return None, None
    assert path is not None
    collector = None if mesh is None else distributed_collector(mesh)
    manager = OutputManager(
        path,
        experiment.output,
        collector=collector,
        attributes=metadata(experiment, conf, phys, mesh),
    )
    observer, _ = output_callbacks(
        manager,
        experiment.step_duration.total_seconds(),
        continue_runs=True,
        start_step=start_step,
        mean_coverage_start_us=mean_coverage_start_us,
        emit_initial_instant=emit_initial_instant,
    )
    assert isinstance(observer, ScheduledObserver)
    return manager, observer


def snapshot(
    path: str | Path,
    state: State,
    experiment: Experiment,
    conf: Configuration,
    phys: PhysicalConstants,
    elapsed: timedelta,
    mesh: Mesh | None,
    variables: Sequence[str] | None,
) -> None:
    """Write selected physical cells at absolute time through the existing schema."""
    _new_path(path)
    settings = experiment.output if experiment.output is not None else OutputSettings()
    collect = storage_fields if mesh is None else distributed_collector(mesh)

    def finite_fields(
        source: State | ArrayFields, names: tuple[str, ...]
    ) -> ArrayFields | None:
        """Validate selected physical fields collectively before the writer opens."""
        values = collect(source, names)
        invalid = (
            -1
            if values is None
            else next(
                (
                    index
                    for index, name in enumerate(names)
                    if not np.isfinite(values[name]).all()
                ),
                -1,
            )
        )
        if mesh is not None:
            from jax.experimental import multihost_utils

            # Only the writer receives collected fields. Share its selected
            # field position before nonwriters can return or open a later
            # collective; every participant reports the same finite failure.
            invalid = int(
                np.asarray(
                    multihost_utils.broadcast_one_to_all(
                        np.asarray(invalid, dtype=np.int32),
                        is_source=jax.process_index() == 0,
                    )
                )
            )
        if invalid >= 0:
            raise ValueError(f"nonfinite output in {names[invalid]}")
        return values

    write_snapshot(
        path,
        state,
        elapsed=elapsed,
        start=settings.start,
        calendar=settings.calendar,
        variables=variables,
        conf=conf,
        phys=phys,
        collector=finite_fields,
        attributes=metadata(experiment, conf, phys, mesh),
    )


def load_fields(
    path: str | Path,
    state: State,
    experiment: Experiment,
    conf: Configuration,
    phys: PhysicalConstants,
    variables: Sequence[str] | None,
    *,
    record: int,
    group: str,
) -> State:
    """Validate a serial selected input completely, then return a ready candidate.

    Supported names are INITIAL_FIELDS and FORCING_FIELDS. Geometry, masks and
    other derived fields require complete reinitialization. Float64 inputs may
    round to float32 when representable; complete restart has stricter semantics.
    Prescribed fields are reapplied by the next absolute physics timestep.
    """
    if conf.use_sharding:
        raise ValueError("selected field input requires serial execution")
    selected = read_record(path, stream=group, index=record, variables=variables)
    if selected.mean:
        raise ValueError("selected field input requires an instantaneous record")
    values = _freeze_fields(selected.fields, INITIAL_FIELDS | FORCING_FIELDS)
    _validate_shapes(values, (experiment.grid.nx, experiment.grid.ny))
    for name, value in values.items():
        _precision(name, value, conf.dtype)
    # Reuse initializer's authoritative derived-quantity range checks with the
    # candidate's actual physical cells, rather than preset defaults or forcing.
    ice = {
        name: values.get(name, getattr(state, name)[2:-2, 2:-2])
        for name in ("hIceMean", "hSnowMean")
    }
    wind = {
        name: values.get(name, getattr(state, name)[2:-2, 2:-2])
        for name in ("uWind", "vWind")
    }
    if "wSpeed" in values:
        wind["wSpeed"] = values["wSpeed"]
    candidate_definition = replace(
        experiment, initial_conditions=ice, forcing=Forcing(constant=wind)
    )
    _validate_derived_inputs(candidate_definition)
    candidate = update_state(state, selected)
    amounts = {"hIceMean", "hSnowMean", "Area"}
    if amounts & values.keys():
        candidate = replace(
            candidate,
            **{
                name: jnp.where(candidate.iceMask != 0, getattr(candidate, name), 0)
                for name in amounts & values.keys()
            },
        )
    for name, mask in (("uIce", "iceMaskU"), ("vIce", "iceMaskV")):
        if name in values:
            candidate = replace(
                candidate,
                **{
                    name: jnp.where(
                        getattr(candidate, mask) != 0, getattr(candidate, name), 0
                    )
                },
            )
    candidate = fill_state_overlap(candidate, conf)
    if {"uWind", "vWind"} & values.keys() and "wSpeed" not in values:
        candidate = replace(
            candidate, wSpeed=jnp.hypot(candidate.uWind, candidate.vWind)
        )
    if amounts & values.keys():
        candidate = replace(
            candidate,
            SeaIceLoad=candidate.hIceMean * phys.rhoIce
            + candidate.hSnowMean * phys.rhoSnow,
            recip_hIceMean=1 / jnp.sqrt(candidate.hIceMean**2 + phys.hIce_reg),
        )
    candidate = fill_state_overlap(candidate, conf)
    jax.block_until_ready(candidate)
    return candidate
