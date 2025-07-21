import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['sett'])
def c_point_to_z_point(vs, sett, Cfield):
    """calculates value at z-point by averaging c-point values"""

    sumNorm = vs.iceMask + jnp.roll(vs.iceMask, 1, 0)
    sumNorm = sumNorm + jnp.roll(sumNorm, 1, 1)
    if sett.noSlip:
        sumNorm = jnp.where(sumNorm > 0, 1.0 / sumNorm, 0.0)
    else:
        sumNorm = jnp.where(sumNorm == 4.0, 0.25, 0.0)

    Zfield = Cfield + jnp.roll(Cfield, 1, 0)
    Zfield = sumNorm * (Zfield + jnp.roll(Zfield, 1, 1))

    return Zfield
