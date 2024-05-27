from veros.core.operators import numpy as npx
from veros import veros_kernel


@veros_kernel
def freedrift_solver(state):
    """calculate ice velocities without taking into account internal ice stress"""

    vs = state.variables
    sett = state.settings

    # air-ice stress at c-point
    tauXIceCenter = 0.5 * (vs.WindForcingX + npx.roll(vs.WindForcingX, -1, 0))
    tauYIceCenter = 0.5 * (vs.WindForcingY + npx.roll(vs.WindForcingY, -1, 1))

    # mass of ice per unit area times coriolis factor
    mIceCor = sett.rhoIce * vs.hIceMean * vs.fCori

    # ocean surface velocity at c-points
    uOceanCenter = 0.5 * (vs.uOcean + npx.roll(vs.uOcean, -1, 0))
    vOceanCenter = 0.5 * (vs.vOcean + npx.roll(vs.vOcean, -1, 1))

    # right hand side of the free drift equation
    rhsX = -tauXIceCenter - mIceCor * vOceanCenter
    rhsY = -tauYIceCenter + mIceCor * uOceanCenter

    # norm of angle of rhs
    tmp1 = rhsX**2 + rhsY**2
    where1 = tmp1 > 0
    rhsN = npx.where(where1, npx.sqrt(rhsX**2 + rhsY**2), 0)
    rhsA = npx.where(where1, npx.arctan2(rhsY, rhsX), 0)

    # solve for norm
    south = vs.fCori < 0
    tmp1 = 1 / (
        npx.where(south, sett.waterIceDrag_south, sett.waterIceDrag) * sett.rhoSea
    )
    tmp2 = tmp1**2 * mIceCor**2
    tmp3 = tmp1**2 * rhsN**2
    tmp4 = tmp2**2 + 4 * tmp3
    solNorm = npx.where(tmp3 > 0, npx.sqrt(0.5 * (npx.sqrt(tmp4) - tmp2)), 0)

    # solve for angle
    tmp1 = 1 / tmp1
    tmp2 = tmp1 * solNorm**2
    tmp3 = mIceCor * solNorm
    tmp4 = tmp2**2 + tmp3**2
    solAngle = npx.where(tmp4 > 0, rhsA - npx.arctan2(tmp3, tmp2), 0)

    # calculate velocities at c-points
    uIceCenter = uOceanCenter - solNorm * npx.cos(solAngle)
    vIceCenter = vOceanCenter - solNorm * npx.sin(solAngle)

    # interpolate to velocity points
    uIceFD = 0.5 * (npx.roll(uIceCenter, 1, 0) + uIceCenter)
    vIceFD = 0.5 * (npx.roll(vIceCenter, 1, 1) + vIceCenter)

    # apply masks
    uIceFD = uIceFD * vs.iceMaskU
    vIceFD = vIceFD * vs.iceMaskV

    return uIceFD, vIceFD
