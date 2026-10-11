"""Versioned JSON definitions and reproducible resolved experiment metadata.

JSON fields use the public global physical (x, y) convention and registry units.
File values override registry defaults; explicit Python overrides merge mappings
by key and replace arrays/lists as whole values. Parsing and serialization stay
on the host, without device selection or allocation. Version 1 stores durations
as integer microseconds and fixed-calendar dates as integer date components.
"""

# ruff: noqa: DTZ001 -- model calendar origins are explicitly timezone-naive.

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import fields
from datetime import datetime, timedelta
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from veris.io.calendar import FixedDate, duration_us
from veris.io.configuration import OUTPUT_SETTINGS, OutputSettings, Stream
from veris.physical_constants import PHYSICALCONSTANTS

if TYPE_CHECKING:
    from veris.experiment import Experiment


def _mapping(value: Any, context: str) -> dict[str, Any]:
    """Require an object with string keys and copy its outer mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object/mapping")
    if any(not isinstance(name, str) for name in value):
        raise TypeError(f"{context} keys must be strings")
    return dict(value)


def _known_keys(data: Mapping[str, Any], names: set[str], context: str) -> None:
    """Identify misspelled options without duplicating constructor validation."""
    unknown = data.keys() - names
    if unknown:
        raise ValueError(f"{context} has unknown keys: {', '.join(sorted(unknown))}")


def _schema_version(version: Any) -> None:
    """Validate format identity independently of overridable experiment values."""
    if type(version) is not int or version != 1:
        raise ValueError("unsupported experiment schema_version; expected version 1")


def _merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested objects while replacing scalar and sequence values entirely."""
    result = dict(base)
    for name, value in overrides.items():
        prior = result.get(name)
        result[name] = (
            _merge(prior, value)
            if isinstance(prior, Mapping) and isinstance(value, Mapping)
            else value
        )
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys at every nesting level during JSON decoding."""
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key {name!r}")
        result[name] = value
    return result


def _nonfinite_constant(value: str) -> Any:
    """Reject JSON extensions that could hide nonfinite scientific inputs."""
    raise ValueError(f"nonfinite JSON constant {value!r} is unsupported")


def read_experiment(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> Experiment:
    """Read version 1 JSON and apply explicit overrides with path-aware errors."""
    source = Path(path)
    try:
        data = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_nonfinite_constant,
        )
        values = _mapping(data, "experiment")
        _schema_version(values.get("schema_version", 1))
        if overrides is not None:
            values = _merge(values, _mapping(overrides, "overrides"))
        return experiment_from_dict(values)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"{source}: {error}") from error


def _duration(value: Any, context: str) -> timedelta:
    """Decode seconds or integer microseconds without losing clock precision."""
    if isinstance(value, timedelta):
        return value
    units = _mapping(value, f"{context} duration")
    if len(units) != 1 or next(iter(units), None) not in (
        "seconds",
        "microseconds",
    ):
        raise ValueError(f"{context} duration requires exactly seconds or microseconds")
    if "microseconds" in units:
        microseconds = units["microseconds"]
        if type(microseconds) is not int:
            raise TypeError(f"{context} duration microseconds must be an integer")
    else:
        seconds = units["seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise TypeError(f"{context} duration seconds must be a real scalar")
        if not math.isfinite(seconds):
            raise ValueError(f"{context} duration seconds must be finite")
        numerator, denominator = Decimal(str(seconds)).as_integer_ratio()
        microseconds, remainder = divmod(numerator * 1_000_000, denominator)
        if remainder:
            raise ValueError(f"{context} duration must use exact integer microseconds")
    try:
        return timedelta(microseconds=microseconds)
    except OverflowError as error:
        raise ValueError(f"{context} duration exceeds the host clock range") from error


def _date(value: Any, calendar: str) -> datetime | FixedDate:
    """Decode Gregorian ISO text or explicit fixed-calendar date components."""
    gregorian = calendar in ("gregorian", "standard", "proleptic_gregorian")
    if isinstance(value, (datetime, FixedDate)):
        return value
    if isinstance(value, str) and gregorian:
        return datetime.fromisoformat(value)
    components = _mapping(value, "output.start date")
    _known_keys(components, {item.name for item in fields(FixedDate)}, "output.start")
    if any(type(component) is not int for component in components.values()):
        raise TypeError("output.start date components must be integers")
    return datetime(**components) if gregorian else FixedDate(**components)


def _output(value: Any) -> OutputSettings | None:
    """Construct output controls using their existing defaults and validators."""
    if value is None or isinstance(value, OutputSettings):
        return value
    values = _mapping(value, "output")
    _known_keys(values, {item.name for item in fields(OutputSettings)}, "output")
    calendar = values.get("calendar", OUTPUT_SETTINGS["calendar"].default)
    if "start" in values:
        values["start"] = _date(values["start"], calendar)
    elif calendar in ("360_day", "noleap", "365_day"):
        default = OUTPUT_SETTINGS["start"].default
        values["start"] = FixedDate(
            **{item.name: getattr(default, item.name) for item in fields(FixedDate)}
        )
    if "streams" in values:
        if not isinstance(values["streams"], (list, tuple)):
            raise TypeError("output.streams must be a sequence of stream objects")
        streams = []
        for index, stream in enumerate(values["streams"]):
            if isinstance(stream, Stream):
                streams.append(stream)
                continue
            context = f"output.streams[{index}]"
            options = _mapping(stream, context)
            _known_keys(options, {item.name for item in fields(Stream)}, context)
            if "sampling_interval" in options:
                options["sampling_interval"] = _duration(
                    options["sampling_interval"], f"{context}.sampling_interval"
                )
            if "period" in options and not isinstance(options["period"], str):
                options["period"] = _duration(options["period"], f"{context}.period")
            streams.append(Stream(**options))
        values["streams"] = tuple(streams)
    return OutputSettings(**values)


def experiment_from_dict(data: Any) -> Experiment:
    """Construct an experiment from grouped, version 1 declarative values.

    The optional schema_version defaults to 1. Numerical and physical options
    retain the validation and defaults of their existing immutable constructors.
    Geometry stores global physical inputs; runnable initialization validates its
    dimensions and scientific metric constraints before allocating a State.
    """
    from veris._typing import OceanGeometry
    from veris.experiment import Execution, Experiment, Grid
    from veris.forcing import Forcing

    values = _mapping(data, "experiment")
    _known_keys(
        values,
        {item.name for item in fields(Experiment) if item.init} | {"schema_version"},
        "experiment",
    )
    _schema_version(values.pop("schema_version", 1))
    if "grid" in values and not isinstance(values["grid"], Grid):
        grid = _mapping(values["grid"], "grid")
        _known_keys(grid, {item.name for item in fields(Grid)}, "grid")
        values["grid"] = Grid(**grid)
    if "execution" in values and not isinstance(values["execution"], Execution):
        execution = _mapping(values["execution"], "execution")
        _known_keys(execution, {item.name for item in fields(Execution)}, "execution")
        if isinstance(execution.get("partitions"), list):
            execution["partitions"] = tuple(execution["partitions"])
        values["execution"] = Execution(**execution)
    for name in ("settings", "constants", "initial_conditions", "scenario"):
        if name in values:
            values[name] = _mapping(values[name], name)
    if "constants" in values:
        values["constants"] = {
            name: tuple(value)
            if name in PHYSICALCONSTANTS
            and PHYSICALCONSTANTS[name].type is tuple
            and isinstance(value, list)
            else value
            for name, value in values["constants"].items()
        }
    if "forcing" in values and not isinstance(values["forcing"], Forcing):
        forcing = _mapping(values["forcing"], "forcing")
        _known_keys(forcing, {"constant", "series"}, "forcing")
        values["forcing"] = Forcing(**forcing)
    if values.get("geometry") is not None and not isinstance(
        values["geometry"], OceanGeometry
    ):
        geometry = _mapping(values["geometry"], "geometry")
        _known_keys(geometry, {item.name for item in fields(OceanGeometry)}, "geometry")
        values["geometry"] = OceanGeometry(**geometry)
    if "output" in values:
        values["output"] = _output(values["output"])
    if values.get("preset") is not None:
        return Experiment.builtin(values.pop("preset"), **values)
    return Experiment(**values)


def _plain(value: Any) -> Any:
    """Return fresh JSON-compatible containers and built-in numeric scalars."""
    if isinstance(value, Mapping):
        return {name: _plain(item) for name, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        if value.dtype.kind == "f":
            value = float(value)
        elif value.dtype.kind in "iu":
            value = int(value)
        else:
            value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("resolved experiment contains an unrepresentable JSON number")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported resolved experiment value {type(value).__name__}")


def _output_dict(output: OutputSettings | None) -> dict[str, Any] | None:
    """Encode exact calendar origins and duration microseconds."""
    if output is None:
        return None
    start = (
        output.start.isoformat(timespec="microseconds")
        if isinstance(output.start, datetime)
        else {item.name: getattr(output.start, item.name) for item in fields(FixedDate)}
    )
    return {
        "start": start,
        "calendar": output.calendar,
        "sample_initial": output.sample_initial,
        "enabled": output.enabled,
        "streams": [
            {
                "name": stream.name,
                "variables": list(stream.variables),
                "sampling_interval": {
                    "microseconds": duration_us(stream.sampling_interval)
                },
                "period": stream.period
                if isinstance(stream.period, str)
                else {"microseconds": duration_us(stream.period)},
            }
            for stream in output.streams
        ],
    }


def experiment_to_dict(
    experiment: Experiment,
    *,
    _array_encoder: Callable[[str, np.ndarray], Any] | None = None,
) -> dict[str, Any]:
    """Serialize resolved constructor defaults and complete grouped host inputs.

    Grid and Execution own dimensions and sharding. Derived numerical/physical
    scalars are recalculated by their constructors rather than serialized as
    override keys. Returned dictionaries and lists do not alias the definition.
    """

    def input_value(path: str, value: Any) -> Any:
        if _array_encoder is not None and isinstance(value, np.ndarray):
            return _array_encoder(path, value)
        return _plain(value)

    def input_fields(path: str, values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: input_value(f"{path}/{name}", value) for name, value in values.items()
        }

    return {
        "schema_version": 1,
        "grid": {"nx": experiment.grid.nx, "ny": experiment.grid.ny},
        "settings": {
            item.name: _plain(getattr(experiment.configuration, item.name))
            for item in fields(experiment.configuration)
            if item.init and item.name not in {"nx", "ny", "use_sharding"}
        },
        "constants": {
            item.name: _plain(getattr(experiment.physical_constants, item.name))
            for item in fields(experiment.physical_constants)
            if item.init
        },
        "physics": experiment.physics,
        "preset": experiment.preset,
        "scenario": _plain(experiment.scenario),
        "geometry": None
        if experiment.geometry is None
        else {
            item.name: input_value(
                f"geometry/{item.name}", getattr(experiment.geometry, item.name)
            )
            for item in fields(experiment.geometry)
        },
        "initial_conditions": input_fields(
            "initial_conditions", experiment.initial_conditions
        ),
        "forcing": {
            "constant": input_fields("forcing/constant", experiment.forcing.constant),
            "series": input_fields("forcing/series", experiment.forcing.series),
        },
        "execution": {
            "backend": experiment.execution.backend,
            "partitions": list(experiment.execution.partitions),
        },
        "steps": experiment.steps,
        "output": _output_dict(experiment.output),
    }
