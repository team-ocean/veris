"""Host allocation of the minimal standalone Veris calculation state.

The VARIABLES registry supplies every field's default, dtype and C-grid
dimensions. Arrays include two halo cells on each boundary, matching the
legacy standalone setup. Physical laws and timestepping choices are separate
immutable objects, so neither adds leaves to the differentiable state.
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.state import State
from veris.variables import VARIABLES


def initialize(
    nx: int | None = None,
    ny: int | None = None,
    *,
    mesh: Mesh | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
    physical_overrides: Mapping[str, Any] | None = None,
    state_overrides: Mapping[str, Any] | None = None,
) -> tuple[State, Settings, PhysicalConstants]:
    """Allocate all fields for an ``nx`` by ``ny`` interior Cartesian grid.

    Omitted extents use Settings defaults or settings_overrides. Explicit nx/ny
    take precedence and are recorded on the returned Settings instance.
    Configuration constructors validate scalar overrides and recompute derived
    values. State overrides must contain numeric arrays with the full storage
    shape, including halos; they are converted to the registry dtype. Each
    interior extent must be at least two cells to supply the periodic halos.
    Enable ``jax_enable_x64`` before calling when metadata requests 64-bit
    arrays; allocation rejects silent dtype truncation. Unknown keys and invalid
    grid extents fail before allocation. Setup-specific forcing and geometry can
    subsequently be applied with dataclasses.replace.

    With an explicit ``mesh``, nx and ny are interior extents per partition.
    Every partition owns two halo cells at each edge; global storage therefore
    has shape ``(mesh.shape['x'] * (nx + 4), mesh.shape['y'] * (ny + 4))``.
    Overrides must already use this packed layout. Arrays are placed with
    NamedSharding(mesh, PartitionSpec('x', 'y')); run sharded kernels inside
    ``jax.set_mesh(mesh)``. Mesh resources remain outside numerical State.
    """
    overrides_settings = dict(settings_overrides or {})
    if nx is not None:
        overrides_settings["nx"] = nx
    if ny is not None:
        overrides_settings["ny"] = ny
    settings = Settings(**overrides_settings)
    sharding = None
    partitions_x = partitions_y = 1
    if mesh is not None:
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a jax.sharding.Mesh")
        if set(mesh.axis_names) != {"x", "y"}:
            raise ValueError("mesh must have axes x and y")
        if not settings.use_sharding:
            raise ValueError("mesh allocation requires use_sharding=True")
        partitions_x, partitions_y = mesh.shape["x"], mesh.shape["y"]
        sharding = NamedSharding(mesh, P("x", "y"))

    for name, metadata in VARIABLES.items():
        if jax.dtypes.canonicalize_dtype(metadata.dtype) != jnp.dtype(metadata.dtype):
            raise ValueError(
                f"{name} requires {metadata.dtype}; enable jax_enable_x64 "
                "before initializing Veris"
            )

    constants = PhysicalConstants(**dict(physical_overrides or {}))
    overrides = dict(state_overrides or {})
    unknown = overrides.keys() - VARIABLES.keys()
    if unknown:
        raise ValueError(f"unknown State fields: {', '.join(sorted(unknown))}")

    dimensions = {
        "x_center": partitions_x * (settings.nx + 4),
        "x_face": partitions_x * (settings.nx + 4),
        "y_center": partitions_y * (settings.ny + 4),
        "y_face": partitions_y * (settings.ny + 4),
    }
    arrays = {}
    for name, metadata in VARIABLES.items():
        shape = tuple(dimensions[dimension] for dimension in metadata.dimensions)
        if name in overrides:
            try:
                array = jnp.asarray(overrides[name], dtype=metadata.dtype)
            except (TypeError, ValueError) as error:
                raise TypeError(f"{name} must be a numeric array") from error
            if array.shape != shape:
                raise ValueError(
                    f"{name} has shape {array.shape}; expected halo-inclusive {shape}"
                )
        else:
            array = jnp.full(shape, metadata.default, dtype=metadata.dtype)
        arrays[name] = (
            jax.device_put(array, sharding) if sharding is not None else array
        )
    return State(**arrays), settings, constants
