import jax.numpy as jnp
from functools import partial
from veris.fill_overlap import fill_overlap_uv
from veris.dynamics_routines import ocean_drag_coeffs


@partial(jax.jit, static_argnames=["sett"]) 
def OceanStressUV(vs, sett):
    """calculate stresses on ocean surface from ocean and ice velocities"""

    # get linear drag coefficient at c-point
    cDrag = ocean_drag_coeffs(vs, sett, vs.uIce, vs.vIce)

    # use turning angle (default is zero)
    sinWat = jnp.sin(jnp.deg2rad(sett.waterTurnAngle))
    cosWat = jnp.cos(jnp.deg2rad(sett.waterTurnAngle))

    # calculate component-wise velocity difference of ice and ocean surface
    du = vs.uIce - vs.uOcean
    dv = vs.vIce - vs.vOcean

    # interpolate to c-points
    duAtC = 0.5 * (du + jnp.roll(du, -1, 0))
    dvAtC = 0.5 * (dv + jnp.roll(dv, -1, 1))

    # calculate forcing on ocean surface in u- and v-direction
    OceanStressU = 0.5 * (cDrag + jnp.roll(cDrag, 1, 0)) * cosWat * du - jnp.sign(
        vs.fCori
    ) * sinWat * 0.5 * (cDrag * dvAtC + jnp.roll(cDrag * dvAtC, 1, 1))
    OceanStressV = 0.5 * (cDrag + jnp.roll(cDrag, 1, 1)) * cosWat * dv + jnp.sign(
        vs.fCori
    ) * sinWat * 0.5 * (cDrag * duAtC + jnp.roll(cDrag * duAtC, 1, 0))

    # fill overlaps
    OceanStressU, OceanStressV = fill_overlap_uv(OceanStressU, OceanStressV)

    return OceanStressU, OceanStressV