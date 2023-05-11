from veros import veros_routine
from veros.core.operators import numpy as npx


@veros_routine
def set_inits(state):
    vs = state.variables

    vs.TSurf = npx.ones_like(vs.maskInC) * 273

    # all other variables are either set in the veros setup file
    # or have 0 as initial value
