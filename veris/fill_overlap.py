from veros.core.operators import update, at
from veros import veros_kernel
from veros.core.utilities import enforce_boundaries


@veros_kernel
def fill_overlap(state, A):

    sett = state.settings

    if sett.veros_fill:
        return enforce_boundaries(A, sett.enable_cyclic_x)
    else:
        A = update(A, at[:sett.olx,:], A[-2*sett.olx:-sett.olx,:])
        A = update(A, at[-sett.olx:,:], A[sett.olx:2*sett.olx,:])
        A = update(A, at[:,:sett.oly], A[:,-2*sett.oly:-sett.oly])
        A = update(A, at[:,-sett.oly:], A[:,sett.oly:2*sett.oly])

        return A

@veros_kernel
def fill_overlap_uv(state, U, V):
    return fill_overlap(state, U), fill_overlap(state, V)