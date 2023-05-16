from veros import veros_routine
from veros.core.operators import numpy as npx


@veros_routine
def set_inits(state):

    vs = state.variables

    # masks
    vs.iceMask = vs.maskT[:, :, -1]
    vs.iceMaskU = vs.maskU[:, :, -1]
    vs.iceMaskV = vs.maskV[:, :, -1]
    vs.maskInC = vs.iceMask
    vs.maskInU = vs.iceMaskU
    vs.maskInV = vs.iceMaskV

    # grid
    vs.R_low = vs.ht
    vs.fCori = vs.coriolis_t
    ones2d = npx.ones_like(vs.maskInC)
    vs.dxC = ones2d * vs.dxt[:, npx.newaxis]
    vs.dyC = ones2d * vs.dyt
    vs.dxU = ones2d * vs.dxu[:, npx.newaxis]
    vs.dyU = ones2d * vs.dyu
    vs.dxG = 0.5 * (vs.dxU + npx.roll(vs.dxU, 1, 1))
    vs.dyG = 0.5 * (vs.dyU + npx.roll(vs.dyU, 1, 0))
    vs.dxV = 0.5 * (vs.dxC + npx.roll(vs.dxC, 1, 1))
    vs.dyV = 0.5 * (vs.dyC + npx.roll(vs.dyC, 1, 0))
    vs.rA = vs.area_t
    vs.rAu = vs.area_u
    vs.rAv = vs.area_v
    vs.rAz = vs.rA + npx.roll(vs.rA, 1, 0)
    vs.rAz = 0.25 * npx.roll(vs.rAz, 1, 1)

    vs.recip_dxC = 1 / vs.dxC
    vs.recip_dyC = 1 / vs.dyC
    vs.recip_dxG = 1 / vs.dxG
    vs.recip_dyG = 1 / vs.dyG
    vs.recip_dxU = 1 / vs.dxU
    vs.recip_dyU = 1 / vs.dyU
    vs.recip_dxV = 1 / vs.dxV
    vs.recip_dyV = 1 / vs.dyV
    vs.recip_rA = 1 / vs.rA
    vs.recip_rAu = 1 / vs.rAu
    vs.recip_rAv = 1 / vs.rAv
    vs.recip_rAz = 1 / vs.rAz

    vs.TSurf = npx.ones_like(vs.maskInC) * 273

    # all other variables are either set in the veros setup file
    # or have 0 as initial value
