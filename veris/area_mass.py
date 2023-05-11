from veros.core.operators import numpy as npx
from veros import veros_kernel


@veros_kernel
def AreaWS(state):
    """calculate sea ice cover fraction centered around velocity points"""

    vs = state.variables

    AreaW = 0.5 * (vs.Area + npx.roll(vs.Area, 1, 0))
    AreaS = 0.5 * (vs.Area + npx.roll(vs.Area, 1, 1))

    return AreaW, AreaS


@veros_kernel
def SeaIceMass(state):
    """calculate mass of the ice-snow system centered around c-, u-, and v-points"""

    vs = state.variables
    sett = state.settings

    SeaIceMassC = sett.rhoIce * vs.hIceMean + sett.rhoSnow * vs.hSnowMean
    SeaIceMassU = 0.5 * (SeaIceMassC + npx.roll(SeaIceMassC, 1, 0))
    SeaIceMassV = 0.5 * (SeaIceMassC + npx.roll(SeaIceMassC, 1, 1))

    return SeaIceMassC, SeaIceMassU, SeaIceMassV
