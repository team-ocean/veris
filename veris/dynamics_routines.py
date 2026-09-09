"""Ice strength, drag, strain, viscosity and stress on the staggered grid."""

from collections.abc import Hashable
from functools import partial
from typing import cast

import jax.numpy as jnp
from jax import Array

from veris._dynamics_types import (
    BasalDragSettings,
    BasalDragState,
    OceanDragSettings,
    OceanDragState,
    SideDragSettings,
    SideDragState,
    StrainSettings,
    StrainState,
    StrengthSettings,
    StrengthState,
    StressDivergenceState,
    ViscositySettings,
    ViscosityState,
)
from veris._typing import ArrayInput, BoundarySettings, MaskState, jit
from veris.physical_constants import PhysicalConstants


@partial(jit, static_argnames=["sett", "phys"])
def SeaIceStrength(
    vs: StrengthState, sett: StrengthSettings, phys: PhysicalConstants
) -> Array:
    """calculate ice strength (= maximum compressive stress)
    from ice thickness and ice cover fraction
    """

    SeaIceStrength = (
        phys.pStar * vs.hIceMean * jnp.exp(-phys.cStar * (1 - vs.Area)) * vs.iceMask
    )

    return SeaIceStrength


@partial(jit, static_argnames=["sett", "phys"])
def ocean_drag_coeffs(
    vs: OceanDragState,
    sett: OceanDragSettings,
    phys: PhysicalConstants,
    uIce: ArrayInput,
    vIce: ArrayInput,
) -> Array:
    """calculate linear ice-water drag coefficient from ice and ocean velocities
    (this coefficient creates a linear relationship between
    ice-ocean stress difference and ice-ocean velocity difference)
    """

    # get ice-water drag coefficient times density
    dragCoeff = (
        jnp.where(vs.fCori < 0, phys.waterIceDrag_south, phys.waterIceDrag)
        * phys.rhoSea
    )

    # calculate component-wise velocity differences at velocity points
    du = (uIce - vs.uOcean) * vs.maskInU
    dv = (vIce - vs.vOcean) * vs.maskInV

    # calculate total velocity difference at c-point
    tmpVar = 0.25 * ((du + jnp.roll(du, -1, 0)) ** 2 + (dv + jnp.roll(dv, -1, 1)) ** 2)

    # calculate linear drag coefficient and apply mask
    cDrag = jnp.where(
        dragCoeff**2 * tmpVar > sett.cDragMin**2,
        dragCoeff * jnp.sqrt(tmpVar),
        sett.cDragMin,
    )
    cDrag = cDrag * vs.iceMask

    return cDrag


@partial(jit, static_argnames=["sett", "phys"])
def basal_drag_coeffs(
    vs: BasalDragState,
    sett: BasalDragSettings,
    phys: PhysicalConstants,
    uIce: ArrayInput,
    vIce: ArrayInput,
) -> Array:
    """calculate basal drag coefficient to account for the formation of
    landfast ice in shallow waters due to the formation of ice keels
    (Lemieux et al., 2015)
    """

    # absolute value of the ice velocity at c-points
    tmpFld = 0.25 * (
        (uIce * vs.maskInU) ** 2
        + jnp.roll(uIce * vs.maskInU, -1, 0) ** 2
        + (vIce * vs.maskInV) ** 2
        + jnp.roll(vIce * vs.maskInV, -1, 1) ** 2
    )

    # include velocity parameter U0 to avoid singularities
    tmpFld = phys.basalDragK2 / jnp.sqrt(tmpFld + phys.basalDragU0**2)

    # critical ice height that allows for the formation of landfast ice
    hCrit = jnp.abs(vs.R_low) * vs.Area / phys.basalDragK1

    # Smooth positive keel excess. logaddexp evaluates log(1 + exp(x))
    # without overflow, including derivatives and masked/disabled drag.
    fac = sett.basalDragSmoothing
    recip_fac = 1.0 / fac
    cBot = jnp.where(
        vs.Area > sett.basalDragMinArea,
        tmpFld
        * jnp.logaddexp(0.0, fac * (vs.hIceMean - hCrit))
        * recip_fac
        * jnp.exp(-phys.cBasalStar * (1.0 - vs.Area)),
        0.0,
    )

    return cBot


@partial(jit, static_argnames=["sett", "phys"])
def side_drag(
    vs: SideDragState,
    sett: SideDragSettings,
    phys: PhysicalConstants,
    uIce: ArrayInput,
    vIce: ArrayInput,
) -> tuple[Array, Array]:
    """calculate the lateral drag coefficient to simulate landfast ice
    (Liu et al. 2022, A new parameterization of coastal drag to simulate landfast
    ice in deep marginal seas in the Arctic)
    """

    # calculate total ice speed at c-points
    iceSpeed = 0.5 * jnp.sqrt(
        (uIce + jnp.roll(uIce, -1, 0)) ** 2 + (vIce + jnp.roll(vIce, -1, 1)) ** 2
    )

    # interpolate total ice speed to u- and v-points
    iceSpeedU = jnp.where(vs.AreaW > 0, 0.5 * (iceSpeed + jnp.roll(iceSpeed, 1, 0)), 0)
    iceSpeedV = jnp.where(vs.AreaS > 0, 0.5 * (iceSpeed + jnp.roll(iceSpeed, 1, 1)), 0)

    # these masks give the number of neighbouring land cells in u- and v-direction
    # (can only be non-zero for ocean cells)
    maskU = (
        2 - jnp.roll(vs.iceMaskU, 1, 0) - jnp.roll(vs.iceMaskU, -1, 0)
    ) * vs.iceMaskU
    maskV = (
        2 - jnp.roll(vs.iceMaskV, 1, 1) - jnp.roll(vs.iceMaskV, -1, 1)
    ) * vs.iceMaskV

    # use the coastline to determine the form factor
    if sett.use_coastline:
        maskU = vs.Fu
        maskV = vs.Fv

    # calculate side drag coefficients
    SideDragU = (
        vs.SeaIceMassU * phys.sideDragCoeff * maskU / (iceSpeedU + phys.sideDragU0)
    )
    SideDragV = (
        vs.SeaIceMassV * phys.sideDragCoeff * maskV / (iceSpeedV + phys.sideDragU0)
    )

    return SideDragU, SideDragV


@partial(jit, static_argnames=["sett", "phys"])
def strainrates(
    vs: StrainState,
    sett: StrainSettings,
    phys: PhysicalConstants,
    uIce: ArrayInput,
    vIce: ArrayInput,
) -> tuple[Array, Array, Array]:
    """calculate strain rate tensor components from ice velocities"""

    # some abbreviations at c-points
    dudx = (jnp.roll(uIce, -1, axis=0) - uIce) * vs.recip_dxU
    uave = (jnp.roll(uIce, -1, axis=0) + uIce) * 0.5
    dvdy = (jnp.roll(vIce, -1, axis=1) - vIce) * vs.recip_dyV
    vave = (jnp.roll(vIce, -1, axis=1) + vIce) * 0.5

    # calculate strain rates at c-points
    e11 = (dudx + vave * vs.k2AtC) * vs.maskInC
    e22 = (dvdy + uave * vs.k1AtC) * vs.maskInC

    # some abbreviations at z-points
    dudy = (uIce - jnp.roll(uIce, 1, axis=1)) * vs.recip_dyU
    uave = (uIce + jnp.roll(uIce, 1, axis=1)) * 0.5
    dvdx = (vIce - jnp.roll(vIce, 1, axis=0)) * vs.recip_dxV
    vave = (vIce + jnp.roll(vIce, 1, axis=0)) * 0.5

    # calculate strain rate at z-points
    mskZ = vs.iceMask * jnp.roll(vs.iceMask, 1, axis=0)
    mskZ = mskZ * jnp.roll(mskZ, 1, axis=1)
    e12 = 0.5 * (dudy + dvdx - vs.k1AtZ * vave - vs.k2AtZ * uave) * mskZ
    if sett.noSlip:
        hFacU = vs.iceMaskU - jnp.roll(vs.iceMaskU, 1, axis=1)
        hFacV = vs.iceMaskV - jnp.roll(vs.iceMaskV, 1, axis=0)
        e12 = e12 + (
            2.0 * uave * vs.recip_dyU * hFacU + 2.0 * vave * vs.recip_dxV * hFacV
        )

    if sett.noSlip and sett.secondOrderBC:
        hFacU = (vs.iceMaskU - jnp.roll(vs.iceMaskU, 1, 0)) / 3.0
        hFacV = (vs.iceMaskV - jnp.roll(vs.iceMaskV, 1, 1)) / 3.0
        hFacU = hFacU * (
            jnp.roll(vs.iceMaskU, 2, 0) * jnp.roll(vs.iceMaskU, 1, 0)
            + jnp.roll(vs.iceMaskU, -1, 0) * vs.iceMaskU
        )
        hFacV = hFacV * (
            jnp.roll(vs.iceMaskV, 2, 1) * jnp.roll(vs.iceMaskV, 1, 1)
            + jnp.roll(vs.iceMaskV, -1, 1) * vs.iceMaskV
        )
        # right hand sided dv/dx = (9*v(i,j)-v(i+1,j))/(4*dxv(i,j)-dxv(i+1,j))
        # according to a Taylor expansion to 2nd order. We assume that dxv
        # varies very slowly, so that the denominator simplifies to 3*dxv(i,j),
        # then dv/dx = (6*v(i,j)+3*v(i,j)-v(i+1,j))/(3*dxv(i,j))
        #            = 2*v(i,j)/dxv(i,j) + (3*v(i,j)-v(i+1,j))/(3*dxv(i,j))
        # the left hand sided dv/dx is analogously
        #            = - 2*v(i-1,j)/dxv(i,j)-(3*v(i-1,j)-v(i-2,j))/(3*dxv(i,j))
        # the first term is the first order part, which is already added.
        # For e12 we only need 0.5 of this gradient and vave = is either
        # 0.5*v(i,j) or 0.5*v(i-1,j) near the boundary so that we need an
        # extra factor of 2. This explains the six. du/dy is analogous.
        # The masking is ugly, but hopefully effective.
        e12 = e12 + 0.5 * (
            vs.recip_dyU
            * (
                6.0 * uave
                - jnp.roll(uIce, 2, 0) * jnp.roll(vs.iceMaskU, 1, 0)
                - jnp.roll(uIce, -1, 0) * vs.iceMaskU
            )
            * hFacU
            + vs.recip_dxV
            * (
                6.0 * vave
                - jnp.roll(vIce, 2, 1) * jnp.roll(vs.iceMaskV, 1, 1)
                - jnp.roll(vIce, -1, 1) * vs.iceMaskV
            )
            * hFacV
        )

    # JIT promotes NumPy inputs to JAX tracers before the stencil runs.
    return e11, e22, cast(Array, e12)


@partial(jit, static_argnames=["sett", "phys"])
def viscosities(
    vs: ViscosityState,
    sett: ViscositySettings,
    phys: PhysicalConstants,
    e11: ArrayInput,
    e22: ArrayInput,
    e12: ArrayInput,
) -> tuple[Array, Array, Array]:
    """calculate bulk viscosity zeta, shear viscosity eta, and ice pressure
    from strain rate tensor components and ice strength.
    if pressReplFac = 1, a replacement pressure is used to avoid
    stresses without velocities (Hibler and Ib, 1995).
    with tensileStrFac != 1, a resistance to tensile stresses can be included
    (König Beatty and Holland, 2010).
    """

    recip_PlasDefCoeffSq = 1.0 / phys.PlasDefCoeff**2

    # interpolate squares of e12 to c-points after weighting them with the
    # area centered around z-points
    e12Csq = vs.rAz * e12**2
    e12Csq = e12Csq + jnp.roll(e12Csq, -1, 0)
    e12Csq = 0.25 * vs.recip_rA * (e12Csq + jnp.roll(e12Csq, -1, 1))

    # calculate Delta from the normal strain rate (e11+e22)
    # and the shear strain rate sqrt( (e11-e22)**2 + 4 * e12**2) )
    deltaSq = (e11 + e22) ** 2 + recip_PlasDefCoeffSq * (
        (e11 - e22) ** 2 + 4.0 * e12Csq
    )
    deltaC = jnp.sqrt(deltaSq)

    # use regularization to avoid singularies of zeta
    deltaCreg = deltaC + sett.deltaMin
    # TODO implement smooth regularization after comparing with the MITgcm
    # smooth regularization of delta for better differentiability
    # deltaCreg = jnp.sqrt( deltaSq + deltaMin**2 )

    # calculate viscosities
    zeta = 0.5 * (vs.SeaIceStrength * (1 + phys.tensileStrFac)) / deltaCreg
    eta = zeta * recip_PlasDefCoeffSq

    # calculate ice pressure
    press = (
        1
        * (
            vs.SeaIceStrength * (1 - sett.pressReplFac)
            + 2.0 * zeta * deltaC * sett.pressReplFac / (1 + phys.tensileStrFac)
        )
        * (1 - phys.tensileStrFac)
    )

    return zeta, eta, press


@partial(jit, static_argnames=["sett", "phys"])
def stress(
    vs: MaskState,
    sett: BoundarySettings,
    phys: PhysicalConstants,
    e11: ArrayInput,
    e22: ArrayInput,
    e12: ArrayInput,
    zeta: ArrayInput,
    eta: ArrayInput,
    press: ArrayInput,
) -> tuple[Array, Array, Array]:
    """calculate stress tensor components"""

    from veris.averaging import c_point_to_z_point

    sig11 = 0.5 * (2 * zeta * (e11 + e22) + 2 * eta * (e11 - e22) - press)
    sig22 = 0.5 * (2 * zeta * (e11 + e22) - 2 * eta * (e11 - e22) - press)
    sig12 = 2.0 * e12 * c_point_to_z_point(vs, sett, phys, eta)

    # NumPy inputs become JAX tracers at this compiled boundary.
    return cast(tuple[Array, Array, Array], (sig11, sig22, sig12))


@partial(jit, static_argnames=["sett", "phys"])
def stressdiv(
    vs: StressDivergenceState,
    sett: Hashable,
    phys: PhysicalConstants,
    sig11: ArrayInput,
    sig22: ArrayInput,
    sig12: ArrayInput,
) -> tuple[Array, Array]:
    """calculate divergence of stress tensor"""

    stressDivX = (
        sig11 * vs.dyV
        - jnp.roll(sig11 * vs.dyV, 1, axis=0)
        - sig12 * vs.dxV
        + jnp.roll(sig12 * vs.dxV, -1, axis=1)
    ) * vs.recip_rAu
    stressDivY = (
        sig22 * vs.dxU
        - jnp.roll(sig22 * vs.dxU, 1, axis=1)
        - sig12 * vs.dyU
        + jnp.roll(sig12 * vs.dyU, -1, axis=0)
    ) * vs.recip_rAv

    # NumPy inputs become JAX tracers at this compiled boundary.
    return cast(tuple[Array, Array], (stressDivX, stressDivY))
