from veros.core.operators import numpy as npx
from veros import veros_kernel


@veros_kernel
def clean_up_advection(state):

    '''clean up overshoots and other pathological cases after advection'''

    vs = state.variables
    sett = state.settings

    # case 1: negative values
    # calculate overshoots of ice and snow thickness
    os_hIceMean = npx.maximum(-vs.hIceMean, 0)
    os_hSnowMean = npx.maximum(-vs.hSnowMean, 0)

    # cut off thicknesses and area at zero
    hIceMean = npx.maximum(vs.hIceMean, 0)
    hSnowMean = npx.maximum(vs.hSnowMean, 0)
    Area = npx.maximum(vs.Area, 0)

    # case 2: very thin ice
    # set thicknesses to zero if the ice thickness is very small
    thinIce = (hIceMean <= sett.hIce_min)
    hIceMean *= ~thinIce
    hSnowMean *= ~thinIce
    TSurf = npx.where(thinIce, sett.celsius2K, vs.TSurf)

    # case 3: area but no ice and snow
    # set area to zero if no ice or snow is present
    Area = npx.where((hIceMean == 0) & (hSnowMean == 0), 0, Area)

    # case 4: very small area
    # introduce lower boundary for the area (if ice or snow is present)
    Area = npx.where((hIceMean > 0) | (hSnowMean > 0),
                        npx.maximum(Area, sett.Area_min), Area)

    return hIceMean, hSnowMean, Area, TSurf, os_hIceMean, os_hSnowMean

@veros_kernel
def ridging(state):

    '''cut off ice cover fraction at 1 after advection to account for ridging'''
    Area = npx.minimum(state.variables.Area, 1)

    return Area