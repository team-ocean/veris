from veros.core.operators import numpy as npx


def c_point_to_z_point(state, Cfield, noSlip=True):
    """calculates value at z-point by averaging c-point values"""

    vs = state.variables

    sumNorm = vs.iceMask + npx.roll(vs.iceMask, 1, 0)
    sumNorm = sumNorm + npx.roll(sumNorm, 1, 1)
    if noSlip:
        sumNorm = npx.where(sumNorm > 0, 1.0 / sumNorm, 0.0)
    else:
        sumNorm = npx.where(sumNorm == 4.0, 0.25, 0.0)

    Zfield = Cfield + npx.roll(Cfield, 1, 0)
    Zfield = sumNorm * (Zfield + npx.roll(Zfield, 1, 1))

    return Zfield
