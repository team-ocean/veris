"""Addressable regional allocation for global physical experiment definitions.

Packed storage owns two halos per partition, while inputs use global physical
indices. Each callback constructs only its addressable region and caches all
70 State fields together. Ocean metrics retain the original adapter arithmetic;
actual global boundary exchange runs after packing, never on each region.
Meshes and distributed communication remain caller-owned host resources.
"""

from collections.abc import Callable, Mapping
from dataclasses import fields
from functools import partial
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.typing import NDArray

from veris.grid.geometry import OceanGeometry
from veris.model.configuration import Configuration
from veris.model.types import State
from veris.model.variables import VARIABLES

type _Indices = NDArray[np.integer[Any]]


def physical_indices(
    index: tuple[slice, ...], conf: Configuration, global_shape: tuple[int, int]
) -> tuple[_Indices, _Indices]:
    """Map a packed storage region to periodic x and periodic/clipped global y."""
    result = []
    for axis, (selection, local, extent) in enumerate(
        zip(index, (conf.nx, conf.ny), global_shape, strict=True)
    ):
        stored = np.arange(selection.start or 0, selection.stop)
        physical = stored // (local + 4) * local + stored % (local + 4) - 2
        result.append(
            physical % extent
            if axis == 0 or conf.enable_cyclic_y
            else np.clip(physical, 0, extent - 1)
        )
    return result[0], result[1]


def regional_geometry(
    geometry: OceanGeometry, x: _Indices, y: _Indices
) -> OceanGeometry:
    """Slice horizontal geometry, spacing vectors and all ocean mask levels."""
    result = {}
    for item in fields(geometry):
        original = getattr(geometry, item.name)
        if original is None or item.name == "radius":
            result[item.name] = original
            continue
        if item.name == "latitude":
            result[item.name] = np.asarray(original)[y[2:-2]]
            continue
        if item.name == "latitude_edges":
            indices = np.r_[y[2:-2], y[-3] + 1]
            result[item.name] = np.asarray(original)[indices]
            continue
        value = np.asarray(original)
        result[item.name] = (
            value[x]
            if item.name in ("dxt", "dxu")
            else value[y]
            if item.name in ("dyt", "dyu")
            else value[x[:, None], y[None, :]]
        )
    return OceanGeometry(**result)


def regional_forcing(value: Any, dtype: str, mesh: Mesh) -> jax.Array:
    """Place host prescribed inputs by addressable callbacks without replication.

    Scalar series retain a whole time axis; spatial series partition only x/y.
    Global host inputs belong to the definition, while each process transfers
    only its addressable physical input regions to devices.
    """
    source = np.asarray(value, dtype=dtype)
    spec = (
        P(None, "x", "y")
        if source.ndim == 3
        else P("x", "y")
        if source.ndim == 2
        else P()
    )
    return jax.make_array_from_callback(
        source.shape,
        NamedSharding(mesh, spec),
        lambda index: source if index is None else source[index],
        dtype=dtype,
    )


def regional_state(
    mesh: Mesh,
    conf: Configuration,
    global_shape: tuple[int, int],
    build: Callable[[_Indices, _Indices], Mapping[str, Any]],
) -> State:
    """Allocate packed State from one cached construction per addressable region."""
    shape = (mesh.shape["x"] * (conf.nx + 4), mesh.shape["y"] * (conf.ny + 4))
    sharding = NamedSharding(mesh, P("x", "y"))
    cache: dict[tuple[int, int, int, int], dict[str, NDArray[Any]]] = {}

    def callback(name: str, index: tuple[slice, ...] | None) -> NDArray[Any]:
        if index is None:
            index = (slice(0, shape[0]), slice(0, shape[1]))
        sx, sy = index
        key = (sx.start or 0, sx.stop or shape[0], sy.start or 0, sy.stop or shape[1])
        if key not in cache:
            x, y = physical_indices(
                (slice(key[0], key[1]), slice(key[2], key[3])), conf, global_shape
            )
            # Regional JAX arithmetic stays on an addressable device outside
            # the global mesh; replicated temporaries would add collectives
            # inside callbacks on a multihost mesh.
            with (
                jax.set_mesh(None),
                jax.default_device(next(iter(sharding.addressable_devices))),
            ):
                values = build(x, y)
                cache[key] = {
                    field: np.asarray(values[field], dtype=conf.dtype)
                    for field in VARIABLES
                }
        return cache[key][name]

    return State(
        **{
            name: jax.make_array_from_callback(
                shape, sharding, partial(callback, name), dtype=conf.dtype
            )
            for name in VARIABLES
        }
    )
