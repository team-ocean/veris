"""Typed host definition arrays for version 1 complete restart files.

The authoritative experiment codec emits references through ArrayManifest;
geometry and physical/time-series inputs remain numeric netCDF variables. No
physics or array allocation takes place here. Logical masks use uint8 storage
and an exact bool descriptor. Paths, dimensions and references validate before
Experiment construction or JAX backend selection.
"""

from dataclasses import fields
from pathlib import Path
from typing import Any

import h5netcdf
import h5py
import numpy as np
from h5py import h5o  # ty: ignore[unresolved-import] -- public compiled HDF object API

from veris.model.experiment import INITIAL_FIELDS
from veris.model.forcing import FORCING_FIELDS
from veris.model.types import OceanGeometry

_SECTIONS = {
    "geometry": {item.name for item in fields(OceanGeometry)},
    "initial_conditions": INITIAL_FIELDS,
    "forcing/constant": FORCING_FIELDS,
    "forcing/series": FORCING_FIELDS,
}


def _allowed(path: str) -> None:
    """Require one known field directly inside one known definition section."""
    section, _, name = path.rpartition("/")
    if section not in _SECTIONS or name not in _SECTIONS[section]:
        raise ValueError(f"invalid definition array reference path {path!r}")


def _group(file: Any, path: str, *, create: bool = False) -> Any:
    """Navigate only validated local group paths, optionally creating them."""
    group = file
    for name in path.split("/"):
        if name not in group.groups:
            if not create:
                raise ValueError(f"missing definition array group {path!r}")
            group.create_group(name)
        group = group.groups[name]
    return group


def validate_links(path: str | Path) -> None:
    """Reject soft/external links and hard aliases before netCDF follows them."""
    with h5py.File(path, "r") as file:
        seen = {h5o.get_info(file["/"].id).addr}

        def visit(group: Any) -> None:
            for name in group:
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise ValueError(f"unsupported HDF link {group.name}/{name}")  # noqa: TRY004 -- on-disk schema corruption
                item = group[name]
                address = h5o.get_info(item.id).addr
                if address in seen:
                    raise ValueError(f"duplicate HDF link {item.name}")
                seen.add(address)
                if isinstance(item, h5py.Group):
                    visit(item)

        visit(file)


def _dtype_descriptor(dtype: np.dtype) -> str:
    """Preserve logical dtype names and non-native numeric byte order."""
    return dtype.name if dtype.isnative else dtype.str


class ArrayManifest:
    """Collect original typed arrays and small references without list copies."""

    def __init__(self) -> None:
        """Start with an empty owned manifest of borrowed immutable inputs."""
        self.arrays: dict[str, np.ndarray] = {}

    def encode(self, path: str, value: np.ndarray) -> dict[str, Any]:
        """Encode one native numeric input using its allowed absolute path."""
        _allowed(path)
        if path in self.arrays:
            raise ValueError(f"duplicate definition array reference {path}")
        if value.dtype.kind not in "bfiu" or not np.isfinite(value).all():
            raise ValueError(f"definition array {path} requires finite real dtype")
        if any(size < 1 for size in value.shape):
            raise ValueError(f"definition array {path} must be nonempty")
        self.arrays[path] = value
        return {
            "array": f"/definition/{path}",
            "shape": list(value.shape),
            "dtype": _dtype_descriptor(value.dtype),
        }

    def write(self, file: h5netcdf.File) -> None:
        """Write independent definition groups after the State writer is closed."""
        for section in _SECTIONS:
            _group(file, f"definition/{section}", create=True)
        for path, value in self.arrays.items():
            section, _, name = path.rpartition("/")
            group = _group(file, f"definition/{section}")
            dimensions = tuple(f"{name}_axis{i}" for i in range(value.ndim))
            group.dimensions.update(dict(zip(dimensions, value.shape, strict=True)))
            storage_dtype = "uint8" if value.dtype.kind == "b" else value.dtype
            variable = group.create_variable(name, dimensions, dtype=storage_dtype)
            variable.attrs["logical_dtype"] = _dtype_descriptor(value.dtype)
            variable[...] = value


def hydrate_definition(
    file: h5netcdf.File, definition: dict[str, Any]
) -> dict[str, Any]:
    """Validate and resolve every reference, rejecting incomplete array groups."""
    # This is a JSON-only tree, so shallow copies of the edited sections suffice.
    result = dict(definition)
    referenced: set[str] = set()
    expected_dimensions: dict[str, dict[str, int]] = {s: {} for s in _SECTIONS}
    for section, allowed_names in _SECTIONS.items():
        names = section.split("/")
        source: Any = definition
        target: Any = result
        for name in names[:-1]:
            source = source.get(name, {})
            if not isinstance(source, dict):
                raise TypeError(f"definition {section} must be an object")
            target[name] = dict(target.get(name, source))
            target = target[name]
        values = source.get(names[-1], {})
        if values is None and section == "geometry":
            continue
        if not isinstance(values, dict):
            raise TypeError(f"definition {section} must be an object")
        hydrated = dict(values)
        target[names[-1]] = hydrated
        for name, reference in values.items():
            if not isinstance(reference, dict):
                continue
            if name not in allowed_names or set(reference) != {
                "array",
                "shape",
                "dtype",
            }:
                raise ValueError(f"invalid definition array reference {section}/{name}")
            path = f"{section}/{name}"
            if reference["array"] != f"/definition/{path}" or path in referenced:
                raise ValueError(
                    f"invalid or duplicate definition array reference {path}"
                )
            shape = reference["shape"]
            if not isinstance(shape, list) or any(
                type(s) is not int or s < 1 for s in shape
            ):
                raise ValueError(f"invalid definition array shape {path}")
            try:
                dtype = np.dtype(reference["dtype"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid definition array dtype {path}") from error
            if (
                dtype.kind not in "bfiu"
                or _dtype_descriptor(dtype) != reference["dtype"]
            ):
                raise ValueError(f"unsupported definition array dtype {path}")
            group = _group(file, f"definition/{section}")
            if name not in group.variables:
                raise ValueError(f"missing definition array {path}")
            variable = group.variables[name]
            dimensions = tuple(f"{name}_axis{i}" for i in range(len(shape)))
            storage_dtype = np.dtype("uint8") if dtype.kind == "b" else dtype
            if (
                variable.dimensions != dimensions
                or variable.shape != tuple(shape)
                or variable.dtype != storage_dtype
                or dict(variable.attrs) != {"logical_dtype": _dtype_descriptor(dtype)}
            ):
                raise ValueError(
                    f"definition array {path} has incompatible dtype/dimensions"
                )
            array = np.asarray(variable[...])
            if not np.isfinite(array).all() or (
                dtype.kind == "b" and np.any(array > 1)
            ):
                raise ValueError(
                    f"definition array {path} requires finite logical values"
                )
            hydrated[name] = array.astype(dtype, copy=False)
            referenced.add(path)
            expected_dimensions[section].update(
                dict(zip(dimensions, shape, strict=True))
            )
    if "definition" not in file.groups:
        raise ValueError("missing definition array group")
    root = file.groups["definition"]
    if (
        set(root.groups) != {"geometry", "initial_conditions", "forcing"}
        or root.variables
        or root.dimensions
        or root.attrs
    ):
        raise ValueError("unexpected definition array schema")
    forcing = root.groups["forcing"]
    if (
        set(forcing.groups) != {"constant", "series"}
        or forcing.variables
        or forcing.dimensions
        or forcing.attrs
    ):
        raise ValueError("unexpected forcing array schema")
    for section in _SECTIONS:
        group = _group(file, f"definition/{section}")
        present = {f"{section}/{name}" for name in group.variables}
        expected = {p for p in referenced if p.rpartition("/")[0] == section}
        if present != expected:
            raise ValueError(f"missing or unreferenced definition arrays in {section}")
        dimensions = {name: len(value) for name, value in group.dimensions.items()}
        if (
            group.groups
            or group.attrs
            or dimensions != expected_dimensions[section]
            or any(value.isunlimited() for value in group.dimensions.values())
        ):
            raise ValueError(
                f"unexpected definition array dimensions/groups in {section}"
            )
    return result
