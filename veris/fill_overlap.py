import jax
import jax.numpy as jnp
import jaxdecomp
from functools import partial


@jax.jit
def fill_circular_overlap(A):
        A = A.at[:2, :].set(A[-4:-2, :])
        A = A.at[-2:, :].set(A[2:4, :])
        A = A.at[:, :2].set(A[:, -4:-2])
        A = A.at[:, -2:].set(A[:, 2:4])

        return A

@partial(jax.jit, static_argnames=['sett'])
def fill_overlap(sett, var):
    if sett.use_circular_overlap:
        return fill_circular_overlap(var)
    else:
        return jaxdecomp.halo_exchange(
                var[:,:,jnp.newaxis], # the jaxdecomp.halo_exchange only works on 3D arrays
                halo_extents=(2, 2), # total halo size in each dimension
                halo_periods=(True, True)
                # True -> periodic/ cyclic halo exchange, the halo values at the left edge
                # of one partition are exchanged with the halo values at the right edge of
                # the adjacent partition. this can be visualized as overlap between partitionings
            )[:,:,0] # remove third axis

@partial(jax.jit, static_argnames=['sett'])
def fill_overlap_uv(sett, u, v):
    return fill_overlap(sett, u), fill_overlap(sett, v)
