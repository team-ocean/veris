"""Periodic two-cell halo exchange for local and mesh-sharded JAX arrays."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
from jax import Array, shard_map
from jax import numpy as jnp
from jax.lax import ppermute
from jax.sharding import AbstractMesh, Mesh
from jax.sharding import PartitionSpec as P

from veris._typing import MaskInput, jit
from veris.configuration import Settings


def fill_circular_overlap(A: Array) -> Array:
    """Copy periodic edges into both horizontal halo regions."""
    A = A.at[:2, :].set(A[-4:-2, :])
    A = A.at[-2:, :].set(A[2:4, :])
    A = A.at[:, :2].set(A[:, -4:-2])
    A = A.at[:, -2:].set(A[:, 2:4])

    return A


def fill_overlap_shard(var: MaskInput) -> Array:
    """runs on each shard, must be inside shard_map"""
    # halo size
    olx, oly = 2, 2

    # get number of devices along both axes
    num_devs_x = int(jax.lax.psum(1, "x"))
    num_devs_y = int(jax.lax.psum(1, "y"))

    # this sends the values of var[-2*olx:-olx,:] from device i to device i+1
    # along the x direction of the processor grid for all processors and
    # stores the received these values (from device i-1 for device i) in left_halo_receive
    left_halo_receive = ppermute(
        var[-2 * olx : -olx, :],
        "x",
        [(i, (i + 1) % num_devs_x) for i in range(num_devs_x)],
    )

    # JAX's arrays cannot be modified in place, therefore functions need to return a new array
    right_halo_receive = ppermute(
        var[olx : 2 * olx, :],
        "x",
        [(i, (i - 1) % num_devs_x) for i in range(num_devs_x)],
    )

    # attach halos in x direction
    var = jnp.concatenate(
        [left_halo_receive, var[olx:-olx, :], right_halo_receive], axis=0
    )

    # exchange and attach halos in y direction
    top_halo_receive = ppermute(
        var[:, -2 * oly : -oly],
        "y",
        [(i, (i + 1) % num_devs_y) for i in range(num_devs_y)],
    )
    bottom_halo_receive = ppermute(
        var[:, oly : 2 * oly],
        "y",
        [(i, (i - 1) % num_devs_y) for i in range(num_devs_y)],
    )
    var = jnp.concatenate(
        [top_halo_receive, var[:, oly:-oly], bottom_halo_receive], axis=1
    )

    return var


def _validate_mesh(mesh: Mesh | AbstractMesh) -> None:
    """Require the two processor-grid axes used by the exchange algorithm."""
    if set(mesh.axis_names) != {"x", "y"}:
        raise ValueError("halo exchange requires an active mesh with axes x and y")


def make_sharded_fill_overlap(mesh: Mesh) -> Callable[[MaskInput], Array]:
    """Bind the existing two-cell exchange to an explicitly supplied device mesh."""
    _validate_mesh(mesh)
    return shard_map(
        fill_overlap_shard, mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
    )


@partial(jit, static_argnames=["sett"])
def fill_overlap(var: MaskInput, sett: Settings) -> Array:
    """Fill periodic halos using initialized settings and the caller's mesh.

    Serial inputs store a single interior with two halo cells on each edge.
    Sharded inputs pack those local halo regions for every mesh partition.
    Sharded inputs use ``NamedSharding(mesh, P("x", "y"))``. Calls must
    execute inside ``jax.set_mesh(mesh)``; mesh execution
    context is owned by the caller and is never stored among State leaves.
    """
    if sett.use_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        _validate_mesh(mesh)
        if {"x", "y"} <= set(mesh.manual_axes):
            # The coupled driver already mapped its entire local stencil.
            return fill_overlap_shard(var)
        fill = shard_map(
            fill_overlap_shard, mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
        )
        return fill(var)
    # JIT converts NumPy inputs to JAX tracers before this body runs.
    return fill_circular_overlap(cast(Array, var))


@partial(jit, static_argnames=["sett"])
def fill_overlap_uv(u: MaskInput, v: MaskInput, sett: Settings) -> tuple[Array, Array]:
    """Fill both horizontal velocity components with the same initialized settings."""
    return fill_overlap(u, sett), fill_overlap(v, sett)
