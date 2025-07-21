import jax
import jaxdecomp

@jax.jit
def fill_overlap(var):
    return jaxdecomp.halo_exchange(
            var[:,:,jnp.newaxis], # the jaxdecomp.halo_exchange only works on 3D arrays
            halo_extents=(2, 2), # total halo size in each dimension
            halo_periods=(True, True)
            # True -> periodic/ cyclic halo exchange, the halo values at the left edge
            # of one partition are exchanged with the halo values at the right edge of
            # the adjacent partition. this can be visualized as overlap between partitionings
        )[:,:,0] # remove third axis

@jax.jit
def fill_overlap_uv(u, v):
    return fill_overlap(u), fill_overlap(v)
