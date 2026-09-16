"""Two-cell halos with periodic x and optional closed global y boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from functools import partial
from typing import Literal, cast

import jax
from jax import Array, shard_map
from jax import numpy as jnp
from jax.lax import ppermute
from jax.sharding import AbstractMesh, Mesh
from jax.sharding import PartitionSpec as P

from veris._typing import MaskInput, State, jit
from veris.configuration import Configuration
from veris.diagnostics import Diagnostics

Boundary = Literal["edge", "zero", "normal", "shear"]


def _close_y(
    var: Array, south: Array | bool, north: Array | bool, boundary: Boundary
) -> Array:
    """Fill exterior y cells and, for normal faces, close the owned south face.

    The C grid stores v at the south face of each tracer cell. Its southern
    wall is index 2; its northern wall is index -2, inside the halo. Only the
    global edge partitions apply these conditions. Edge extension keeps scalar
    temperatures and reciprocal metrics valid outside the physical domain.
    """
    lower = var[:, 2:3] if boundary == "edge" else jnp.zeros_like(var[:, :1])
    upper = var[:, -3:-2] if boundary == "edge" else jnp.zeros_like(var[:, :1])
    if boundary == "shear":
        # Unlike normal velocity, no-slip wall traction is a calculated value.
        # Its northern physical face is stored in the first halo column.
        upper = jnp.concatenate((var[:, -2:-1], jnp.zeros_like(var[:, :1])), axis=1)
    var = var.at[:, :2].set(jnp.where(south, lower, var[:, :2]))
    var = var.at[:, -2:].set(jnp.where(north, upper, var[:, -2:]))
    if boundary == "normal":
        var = var.at[:, 2].set(jnp.where(south, 0, var[:, 2]))
    return var


def fill_circular_overlap(
    A: Array, enable_cyclic_y: bool = True, boundary: Boundary = "edge"
) -> Array:
    """Fill periodic x halos and periodic or closed global y halos."""
    A = A.at[:2, :].set(A[-4:-2, :])
    A = A.at[-2:, :].set(A[2:4, :])
    if enable_cyclic_y:
        A = A.at[:, :2].set(A[:, -4:-2])
        A = A.at[:, -2:].set(A[:, 2:4])
    else:
        A = _close_y(A, True, True, boundary)

    return A


def fill_overlap_shard(
    var: MaskInput, enable_cyclic_y: bool = True, boundary: Boundary = "edge"
) -> Array:
    """runs on each shard, must be inside shard_map"""
    # halo size
    olx, oly = 2, 2

    # get number of devices along both axes
    num_devs_x = int(jax.lax.psum(1, "x"))
    num_devs_y = int(jax.lax.psum(1, "y"))
    if not enable_cyclic_y and boundary == "normal":
        # On a two-cell partition the south wall face is also sent to the
        # next partition's halo. Close the owned value before communicating.
        rank_y = jax.lax.axis_index("y")
        var = _close_y(
            jnp.asarray(var), rank_y == 0, rank_y == num_devs_y - 1, boundary
        )

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
    north_shear = var[:, -2:-1]

    # exchange and attach halos in y direction
    top_halo_receive = ppermute(
        var[:, -2 * oly : -oly],
        "y",
        [
            (i, (i + 1) % num_devs_y)
            for i in range(num_devs_y)
            if enable_cyclic_y or i + 1 < num_devs_y
        ],
    )
    bottom_halo_receive = ppermute(
        var[:, oly : 2 * oly],
        "y",
        [
            (i, (i - 1) % num_devs_y)
            for i in range(num_devs_y)
            if enable_cyclic_y or i > 0
        ],
    )
    var = jnp.concatenate(
        [top_halo_receive, var[:, oly:-oly], bottom_halo_receive], axis=1
    )

    if not enable_cyclic_y:
        rank_y = jax.lax.axis_index("y")
        if boundary == "shear":
            var = var.at[:, -2:-1].set(
                jnp.where(rank_y == num_devs_y - 1, north_shear, var[:, -2:-1])
            )
        var = _close_y(var, rank_y == 0, rank_y == num_devs_y - 1, boundary)
    return var


def _validate_mesh(mesh: Mesh | AbstractMesh) -> None:
    """Require the two processor-grid axes used by the exchange algorithm."""
    if set(mesh.axis_names) != {"x", "y"}:
        raise ValueError("halo exchange requires an active mesh with axes x and y")


def make_sharded_fill_overlap(
    mesh: Mesh, *, enable_cyclic_y: bool = True, boundary: Boundary = "edge"
) -> Callable[[MaskInput], Array]:
    """Bind the existing two-cell exchange to an explicitly supplied device mesh."""
    _validate_mesh(mesh)
    return shard_map(
        partial(fill_overlap_shard, enable_cyclic_y=enable_cyclic_y, boundary=boundary),
        mesh=mesh,
        in_specs=P("x", "y"),
        out_specs=P("x", "y"),
    )


@partial(jit, static_argnames=["conf", "boundary"])
def fill_overlap(
    var: MaskInput, conf: Configuration, *, boundary: Boundary = "edge"
) -> Array:
    """Fill halos using initialized settings and the caller's mesh.

    Serial inputs store a single interior with two halo cells on each edge.
    Sharded inputs pack those local halo regions for every mesh partition.
    Sharded inputs use ``NamedSharding(mesh, P("x", "y"))``. Calls must
    execute inside ``jax.set_mesh(mesh)``; mesh execution
    context is owned by the caller and is never stored among State leaves.
    Closed y boundaries use nearest-edge extension by default; ``zero`` marks
    dry exterior cells and ``normal`` also closes the southern wall face.
    ``shear`` preserves both physical wall tractions, including the north halo
    face, while zeroing exterior cells beyond the walls.
    Periodic mode ignores these wall policies.
    """
    if boundary not in ("edge", "zero", "normal", "shear"):
        raise ValueError("boundary must be edge, zero, normal or shear")
    exchange = partial(
        fill_overlap_shard, enable_cyclic_y=conf.enable_cyclic_y, boundary=boundary
    )
    if conf.use_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        _validate_mesh(mesh)
        if {"x", "y"} <= set(mesh.manual_axes):
            # The coupled driver already mapped its entire local stencil.
            return exchange(var)
        fill = shard_map(
            exchange, mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
        )
        return fill(var)
    # JIT converts NumPy inputs to JAX tracers before this body runs.
    return fill_circular_overlap(cast(Array, var), conf.enable_cyclic_y, boundary)


@partial(jit, static_argnames=["conf"])
def fill_overlap_uv(
    u: MaskInput, v: MaskInput, conf: Configuration
) -> tuple[Array, Array]:
    """Fill both horizontal velocity components with the same initialized settings."""
    return fill_overlap(u, conf, boundary="zero"), fill_overlap(
        v, conf, boundary="normal"
    )


def fill_state_overlap[T: (State, Diagnostics)](state: T, conf: Configuration) -> T:
    """Refresh State or Diagnostics halos with impermeable-wall field semantics.

    Dry exterior masks activate the existing coastline/slip discretization.
    Ice amounts and tangential velocities vanish outside the domain; normal
    velocities also vanish on the owned southern wall face. Other fields use
    constant extension, avoiding invalid zero temperatures and grid metrics.
    Corner shear stress retains no-slip wall traction or vanishes for free slip.
    """
    zero = {
        "iceMask",
        "iceMaskU",
        "maskInC",
        "maskInU",
        "uIce",
        "uOcean",
        "hIceMean",
        "hSnowMean",
        "Area",
        "OceanStressU",
    }
    normal = {"iceMaskV", "maskInV", "vIce", "vOcean", "OceanStressV"}
    return replace(
        state,
        **{
            field.name: fill_overlap(
                getattr(state, field.name),
                conf,
                boundary=("shear" if conf.noSlip else "normal")
                if field.name == "sigma12"
                else "normal"
                if field.name in normal
                else "zero"
                if field.name in zero
                else "edge",
            )
            for field in fields(state)
        },
    )
