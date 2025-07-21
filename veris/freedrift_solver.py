import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['sett'])
def freedrift_solver(vs, sett):
    """calculate ice velocities without taking into account internal ice stress"""

    # air-ice stress at c-point
    tauXIceCenter = 0.5 * (vs.WindForcingX + jnp.roll(vs.WindForcingX, -1, 0))
    tauYIceCenter = 0.5 * (vs.WindForcingY + jnp.roll(vs.WindForcingY, -1, 1))

    # mass of ice per unit area times coriolis factor
    mIceCor = sett.rhoIce * vs.hIceMean * vs.fCori

    # ocean surface velocity at c-points
    uOceanCenter = 0.5 * (vs.uOcean + jnp.roll(vs.uOcean, -1, 0))
    vOceanCenter = 0.5 * (vs.vOcean + jnp.roll(vs.vOcean, -1, 1))

    # right hand side of the free drift equation
    rhsX = -tauXIceCenter - mIceCor * vOceanCenter
    rhsY = -tauYIceCenter + mIceCor * uOceanCenter

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    where1 = tmp1 > 0
    rhsN = jnp.where(where1, jnp.sqrt(tmp1), 0)
    rhsA = jnp.where(where1, jnp.arctan2(rhsY, rhsX), 0)

    # solve for norm
    south = vs.fCori < 0
    tmp1 = 1 / (
        jnp.where(south, sett.waterIceDrag_south, sett.waterIceDrag) * sett.rhoSea
    )
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = jnp.where(tmp3 > 0, jnp.sqrt(0.5 * (jnp.sqrt(tmp4) - tmp2)), 0)

    # solve for angle
    tmp1 = 1 / tmp1
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = jnp.where(tmp4 > 0, rhsA - jnp.arctan2(tmp3, tmp2), 0)

    # calculate velocities at c-points
    uIceCenter = uOceanCenter - solNorm * jnp.cos(solAngle)
    vIceCenter = vOceanCenter - solNorm * jnp.sin(solAngle)

    # interpolate to velocity points
    uIceFD = 0.5 * (jnp.roll(uIceCenter, 1, 0) + uIceCenter)
    vIceFD = 0.5 * (jnp.roll(vIceCenter, 1, 1) + vIceCenter)

    # apply masks
    uIceFD = uIceFD * vs.iceMaskU
    vIceFD = vIceFD * vs.iceMaskV

    return uIceFD, vIceFD
