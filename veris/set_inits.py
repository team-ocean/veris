"""Initialize sea-ice staggered grid fields from surface ocean geometry.

This host routine updates a mutable state.variables container with JAX arrays.
Input ocean masks have shape (nx, ny, nz); horizontal metrics are 1-D or 2-D.
The last vertical mask level is the ocean surface. Periodic neighbor averages
follow the original Veris set_inits routine.
"""

from typing import Protocol

import jax.numpy as npx
from jax import Array


class MutableGeometry(Protocol):
    """Ocean geometry inputs and writable output slots populated by set_inits.

    Output slots may be uninitialized on entry; every output is assigned before
    use. Inputs are surface/volume JAX arrays with the dimensions in set_inits.
    """

    @property
    def area_t(self) -> Array: ...

    @property
    def area_u(self) -> Array: ...

    @property
    def area_v(self) -> Array: ...

    @property
    def coriolis_t(self) -> Array: ...

    @property
    def dxt(self) -> Array: ...

    @property
    def dxu(self) -> Array: ...

    @property
    def dyt(self) -> Array: ...

    @property
    def dyu(self) -> Array: ...

    @property
    def ht(self) -> Array: ...

    @property
    def maskT(self) -> Array: ...

    @property
    def maskU(self) -> Array: ...

    @property
    def maskV(self) -> Array: ...

    # Mutable horizontal outputs created by initialization.
    R_low: Array
    TSurf: Array
    dxC: Array
    dxG: Array
    dxU: Array
    dxV: Array
    dyC: Array
    dyG: Array
    dyU: Array
    dyV: Array
    fCori: Array
    iceMask: Array
    iceMaskU: Array
    iceMaskV: Array
    maskInC: Array
    maskInU: Array
    maskInV: Array
    rA: Array
    rAu: Array
    rAv: Array
    rAz: Array
    recip_dxC: Array
    recip_dxG: Array
    recip_dxU: Array
    recip_dxV: Array
    recip_dyC: Array
    recip_dyG: Array
    recip_dyU: Array
    recip_dyV: Array
    recip_rA: Array
    recip_rAu: Array
    recip_rAv: Array
    recip_rAz: Array


class GeometryState(Protocol):
    """Host container exposing geometry storage, distinct from immutable ice state."""

    @property
    def variables(self) -> MutableGeometry: ...


def set_inits(state: GeometryState) -> None:
    """Populate surface masks, staggered areas/metrics and their reciprocals."""

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
    vs.rAz = 0.25 * (vs.rAz + npx.roll(vs.rAz, 1, 1))

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

    # The caller initializes physical ice, ocean, and atmospheric fields.
