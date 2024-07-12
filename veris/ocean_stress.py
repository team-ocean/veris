from veros.core.operators import numpy as npx
from veros import veros_kernel

from veris.fill_overlap import fill_overlap_uv
from veris.dynamics_routines import ocean_drag_coeffs


@veros_kernel
def OceanStressUV(state):
    """calculate stresses on ocean surface from ocean and ice velocities"""

    vs = state.variables
    sett = state.settings

    # get linear drag coefficient at c-point
    cDrag = ocean_drag_coeffs(state, vs.uIce, vs.vIce)

    # use turning angle (default is zero)
    sinWat = npx.sin(npx.deg2rad(sett.waterTurnAngle))
    cosWat = npx.cos(npx.deg2rad(sett.waterTurnAngle))

    # calculate component-wise velocity difference of ice and ocean surface
    du = vs.uIce - vs.uOcean
    dv = vs.vIce - vs.vOcean

    # interpolate to c-points
    duAtC = 0.5 * (du + npx.roll(du, -1, 0))
    dvAtC = 0.5 * (dv + npx.roll(dv, -1, 1))

    # calculate forcing on ocean surface in u- and v-direction
    OceanStressU = 0.5 * (cDrag + npx.roll(cDrag, 1, 0)) * cosWat * du - npx.sign(
        vs.fCori
    ) * sinWat * 0.5 * (cDrag * dvAtC + npx.roll(cDrag * dvAtC, 1, 1))
    OceanStressV = 0.5 * (cDrag + npx.roll(cDrag, 1, 1)) * cosWat * dv + npx.sign(
        vs.fCori
    ) * sinWat * 0.5 * (cDrag * duAtC + npx.roll(cDrag * duAtC, 1, 0))

    # fill overlaps
    OceanStressU, OceanStressV = fill_overlap_uv(state, OceanStressU, OceanStressV)

    return OceanStressU, OceanStressV