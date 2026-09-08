from functools import partial
from importlib import import_module

import jax
from jax import numpy as jnp
from jax import shard_map
from jax.lax import ppermute
from jax.sharding import PartitionSpec as P

from veris.settings import settings


def fill_circular_overlap(A):
    A = A.at[:2, :].set(A[-4:-2, :])
    A = A.at[-2:, :].set(A[2:4, :])
    A = A.at[:, :2].set(A[:, -4:-2])
    A = A.at[:, -2:].set(A[:, 2:4])

    return A


def fill_overlap_shard(var):
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


def make_sharded_fill_overlap():
    """return a shard_map-wrapped version of fill_overlap for the initialized mesh"""
    mesh = import_module("initialize_mesh_sharding").mesh
    if mesh is None:
        raise RuntimeError("mesh and sharding not initialized")
    return shard_map(
        fill_overlap_shard, mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
    )


if settings["use_sharding"]:
    """use the correct fill_overlap function, depending on whether
    veris is run is a distributed runtime with sharded arrays or not
    """
    sharded_fill_overlap = make_sharded_fill_overlap()

    @partial(jax.jit)
    def fill_overlap(var):
        return sharded_fill_overlap(var)
else:

    @partial(jax.jit)
    def fill_overlap(var):
        return fill_circular_overlap(var)


@partial(jax.jit)
def fill_overlap_uv(u, v):
    return fill_overlap(u), fill_overlap(v)
