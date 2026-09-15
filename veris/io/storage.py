"""Host-side h5netcdf records and validated State snapshot input.

Arrays use the VARIABLES registry's (x, y) C-grid dimensions. File records are
append-only, selected-field inputs update an already initialized State, and
full halo-inclusive snapshots preserve its arrays. No numerical kernels or
source-reference equations are changed by this storage layer.
"""

import json
import re

# ruff: noqa: DTZ001 -- model dates intentionally have no timezone.
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import h5netcdf
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from veris._typing import State
from veris.configuration import Configuration
from veris.io.calendar import Calendar, FixedDate, duration_us
from veris.io.guard import require_host
from veris.physical_constants import PhysicalConstants
from veris.variables import VARIABLES

ArrayFields = Mapping[str, Any]
Collector = Callable[
    [State | ArrayFields, tuple[str, ...], bool], Mapping[str, Any] | None
]


def selected_fields(
    source: State | ArrayFields,
    names: Sequence[str] | None = None,
    *,
    include_halos: bool = True,
) -> dict[str, NDArray[Any]]:
    """Validate and materialize selected arrays; optionally trim serial halos."""
    require_host()
    if names is None:
        names = tuple(source) if isinstance(source, Mapping) else tuple(VARIABLES)
    names = tuple(names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in VARIABLES for n in names)
    ):
        raise ValueError("variables must be a nonempty unique selection from VARIABLES")
    result = {}
    shape = None
    for name in names:
        if (isinstance(source, Mapping) and name not in source) or (
            not isinstance(source, Mapping) and not hasattr(source, name)
        ):
            raise ValueError(f"missing variable {name}")
        value = source[name] if isinstance(source, Mapping) else getattr(source, name)
        if getattr(getattr(value, "sharding", None), "num_devices", 1) > 1:
            raise ValueError(
                "multi-device State output requires a distributed collector"
            )
        array = np.asarray(value)
        if array.ndim != 2 or array.dtype.kind not in "fiu" or min(array.shape) < 1:
            raise ValueError(
                f"variable {name} must be a nonempty real numeric 2D array"
            )
        if shape is not None and shape != array.shape:
            raise ValueError(f"variable {name} has inconsistent grid shape")
        shape = array.shape
        if not include_halos:
            if min(array.shape) <= 4:
                raise ValueError(
                    f"variable {name} has no interior beyond two-cell halos"
                )
            array = array[2:-2, 2:-2]
        result[name] = array
    return result


class NetCDFWriter:
    """Lazily create a new netCDF file and append records to independent groups."""

    def __init__(
        self,
        path: str | Path,
        *,
        calendar: str,
        units: str,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.calendar = calendar
        self.units = units
        self.attributes = dict(attributes or {})
        self._file: Any = None
        self._closed = False

    def __enter__(self) -> Self:
        """Return the lazy writer; no file exists until the first record."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file on both normal and exceptional exits."""
        self.close()

    def validate_schema(
        self,
        stream: str,
        fields: ArrayFields,
        *,
        mean: bool = False,
        include_halos: bool = False,
        time: float | None = None,
        mean_samples: bool = False,
    ) -> None:
        """Check an existing stream before any manager or file state changes.

        mean_samples describes incoming samples that the manager will accumulate
        into float64 means; their storage dtype is therefore float64.
        """
        require_host()
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._file is None:
            if self.path.exists():
                raise FileExistsError(self.path)
            return
        if stream not in self._file.groups:
            return
        group = self._file.groups[stream]
        arrays = fields
        expected = {name for name in group.variables if name in VARIABLES}
        if (
            expected != set(arrays)
            or bool(group.attrs["mean"]) != mean
            or bool(group.attrs["include_halos"]) != include_halos
        ):
            raise ValueError("stream variables and record kind cannot change")
        for name, array in arrays.items():
            if group.variables[name].shape[1:] != array.shape:
                raise ValueError(f"variable {name} changed shape")
            if not np.can_cast(
                np.dtype("float64") if mean_samples else array.dtype,
                group.variables[name].dtype,
                casting="safe",
            ):
                raise ValueError(f"variable {name} changed to an unsafe dtype")
        index = len(group.dimensions["time"])
        if index and time is not None and time <= group.variables["time"][index - 1]:
            raise ValueError("record times must increase")

    def append(
        self,
        stream: str,
        fields: ArrayFields,
        *,
        time: float,
        bounds: tuple[float, float],
        count: int = 1,
        partial: bool = False,
        mean: bool = False,
        include_halos: bool = False,
    ) -> None:
        """Append one validated record with equal-weight sample count and bounds."""
        require_host()
        if self._closed:
            raise RuntimeError("writer is closed")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", stream):
            raise ValueError("stream name must be a simple netCDF identifier")
        arrays = selected_fields(fields)
        if (
            not np.isfinite((time, *bounds)).all()
            or not bounds[0] <= time <= bounds[1]
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
        ):
            raise ValueError(
                "record requires finite ordered time bounds and positive sample count"
            )
        self.validate_schema(
            stream, arrays, mean=mean, include_halos=include_halos, time=time
        )
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = h5netcdf.File(self.path, "x")
            self._file.attrs.update({"Conventions": "CF-1.10", **self.attributes})
        file = self._file
        if stream not in file.groups:
            group = file.create_group(stream)
            group.dimensions = {"time": None, "bounds": 2}
            group.attrs.update({"include_halos": int(include_halos), "mean": int(mean)})
            for name, array in arrays.items():
                for dim, size in zip(
                    VARIABLES[name].dimensions, array.shape, strict=True
                ):
                    if dim not in group.dimensions:
                        group.dimensions[dim] = size
                variable = group.create_variable(
                    name, ("time", *VARIABLES[name].dimensions), dtype=array.dtype
                )
                variable.attrs.update(VARIABLES[name].netcdf_attributes())
                variable.attrs["cell_methods"] = "time: mean" if mean else "time: point"
            coordinate = group.create_variable("time", ("time",), dtype="f8")
            coordinate.attrs.update(
                {
                    "units": self.units,
                    "calendar": self.calendar,
                    "bounds": "time_bounds",
                    "standard_name": "time",
                }
            )
            group.create_variable("time_bounds", ("time", "bounds"), dtype="f8")
            group.create_variable("sample_count", ("time",), dtype="i8")
            group.create_variable("partial", ("time",), dtype="i1")
        group = file.groups[stream]
        index = len(group.dimensions["time"])
        group.resize_dimension("time", index + 1)
        for name, array in arrays.items():
            group.variables[name][index] = array
        for name, value in {
            "time": time,
            "time_bounds": bounds,
            "sample_count": count,
            "partial": int(partial),
        }.items():
            group.variables[name][index] = value
        file.flush()

    def close(self) -> None:
        """Close a file if created; safe to call repeatedly outside tracing."""
        require_host()
        if self._file is not None:
            self._file.close()
            self._file = None
        self._closed = True


@dataclass(frozen=True)
class Record:
    """One materialized record plus calendar, coverage and snapshot metadata."""

    fields: dict[str, NDArray[Any]]
    time: float
    bounds: tuple[float, float]
    count: int
    partial: bool
    mean: bool
    include_halos: bool
    calendar: str
    units: str
    configuration: dict[str, Any]
    physical_constants: dict[str, Any]


def read_record(
    path: str | Path,
    *,
    stream: str = "snapshot",
    index: int = -1,
    variables: Sequence[str] | None = None,
) -> Record:
    """Read a selected record, rejecting unknown fields and incompatible dimensions."""
    require_host()
    with h5netcdf.File(path, "r") as file:
        if stream not in file.groups:
            raise ValueError(f"unknown stream {stream}")
        group = file.groups[stream]
        available = tuple(name for name in group.variables if name in VARIABLES)
        names = tuple(variables) if variables is not None else available
        if (
            not names
            or len(set(names)) != len(names)
            or any(n not in available for n in names)
        ):
            raise ValueError("unknown or duplicate variable selection")
        size = len(group.dimensions["time"])
        if not -size <= index < size:
            raise IndexError("record index outside time dimension")
        arrays = {}
        for name in names:
            variable = group.variables[name]
            if variable.dimensions != ("time", *VARIABLES[name].dimensions):
                raise ValueError(f"variable {name} has incompatible dimensions")
            arrays[name] = np.asarray(variable[index])
        arrays = selected_fields(arrays)
        return Record(
            arrays,
            float(group.variables["time"][index]),
            (
                float(group.variables["time_bounds"][index, 0]),
                float(group.variables["time_bounds"][index, 1]),
            ),
            int(group.variables["sample_count"][index]),
            bool(group.variables["partial"][index]),
            bool(group.attrs["mean"]),
            bool(group.attrs["include_halos"]),
            str(group.variables["time"].attrs["calendar"]),
            str(group.variables["time"].attrs["units"]),
            json.loads(file.attrs.get("configuration", "{}")),
            json.loads(file.attrs.get("physical_constants", "{}")),
        )


def update_state(state: State, record: Record) -> State:
    """Apply halo-inclusive instantaneous fields to an initialized compatible State."""
    require_host()
    if record.mean:
        raise ValueError("a mean record is not an instantaneous State snapshot")
    if not record.include_halos:
        raise ValueError("State updates require halo-inclusive snapshots")
    arrays = selected_fields(record.fields)
    for name, value in arrays.items():
        if value.shape != getattr(state, name).shape:
            raise ValueError(f"variable {name} shape does not match initialized State")
    return replace(
        state,
        **{
            name: jnp.asarray(value, dtype=getattr(state, name).dtype)
            for name, value in arrays.items()
        },
    )


def write_snapshot(
    path: str | Path,
    state: State | ArrayFields,
    *,
    elapsed: timedelta = timedelta(0),
    start: datetime | FixedDate = datetime(2000, 1, 1),
    calendar: str = "gregorian",
    variables: Sequence[str] | None = None,
    include_halos: bool = True,
    conf: Configuration | None = None,
    phys: PhysicalConstants | None = None,
    collector: Collector | None = None,
) -> None:
    """Write a concrete final State outside integration/AD, independently of streams."""
    require_host()
    clock = Calendar(start, calendar)
    clock.date_at(elapsed)
    if collector is None:
        if conf is not None and conf.use_sharding:
            raise ValueError("sharded State output requires a distributed collector")
        fields = selected_fields(state, variables, include_halos=include_halos)
    else:
        names = (
            tuple(variables)
            if variables is not None
            else (tuple(state) if isinstance(state, Mapping) else tuple(VARIABLES))
        )
        if (
            not names
            or len(set(names)) != len(names)
            or any(name not in VARIABLES for name in names)
        ):
            raise ValueError("invalid variable selection")
        collected = collector(state, names, include_halos)
        if collected is None:
            return
        fields = selected_fields(collected, names)
    attributes = {}
    if conf is not None:
        attributes["configuration"] = json.dumps(asdict(conf))
    if phys is not None:
        attributes["physical_constants"] = json.dumps(asdict(phys))
    time = duration_us(elapsed) / 1e6
    with NetCDFWriter(
        path, calendar=clock.name, units=clock.units, attributes=attributes
    ) as writer:
        writer.append(
            "snapshot",
            fields,
            time=time,
            bounds=(time, time),
            include_halos=include_halos,
        )
