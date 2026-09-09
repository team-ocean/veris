"""Solve the original Veris conductive and atmospheric ice/snow heat balance.

Horizontal thickness, temperature, and forcing arrays produce surface
temperature, two net heat fluxes, penetrating shortwave radiation, and
sublimation. Six Newton iterations retain the reference melting-temperature
cap and albedo switches; gradients follow their selected smooth branches.
"""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array

from veris._thermodynamic_types import SurfaceFluxResult
from veris._typing import ArrayInput, jit
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.state import State


@partial(jit, static_argnames=["sett", "phys"])
def solve4temp(
    vs: State,
    sett: Settings,
    phys: PhysicalConstants,
    hIceActual: ArrayInput,
    hSnowActual: ArrayInput,
    TSurfIn: ArrayInput,
    TempFrz: ArrayInput,
) -> SurfaceFluxResult:
    """calculate heat fluxes through the ice and ice surface temperature"""

    ##### define local constants used for calculations #####

    # coefficients for the saturation vapor pressure equation
    aa1 = phys.iceVaporPressureTemperature
    aa2 = phys.iceVaporPressureLog10Offset
    bb1 = phys.waterVaporDryAirMassRatio
    bb2 = 1 - bb1
    Ppascals = phys.iceSurfacePressure
    cc0 = 10**aa2
    cc1 = cc0 * aa1 * bb1 * Ppascals * jnp.log(10)
    cc2 = cc0 * bb2

    # sensible heat constant
    d1 = phys.dalton * phys.cpAir * phys.rhoAir
    # latent heat constant
    d1i = phys.dalton * phys.lhSublim * phys.rhoAir

    # melting temperature of ice
    Tmelt = phys.celsius2K

    # temperature threshold for when to use wet albedo
    SurfMeltTemp = Tmelt + phys.wetAlbTemp

    # make local copies of downward longwave radiation, surface
    # and atmospheric temperatures
    TSurfLoc = TSurfIn
    LWdownLocCapped = jnp.maximum(sett.minLWdown, vs.LWdown)
    ATempLoc = jnp.maximum(phys.celsius2K + sett.minTAir, vs.ATemp)

    # set wind speed with lower boundary
    ug = jnp.maximum(sett.wSpeedMin, vs.wSpeed)

    isIce = hIceActual > 0
    isSnow = hSnowActual > 0

    d3 = jnp.where(isSnow, phys.snowEmiss, phys.iceEmiss) * phys.stefBoltz

    LWdownLoc = jnp.where(isSnow, phys.snowEmiss, phys.iceEmiss) * LWdownLocCapped

    ##### determine albedo #####

    # use albedo of dry surface (if ice is present)
    albIce = jnp.where(isIce, phys.dryIceAlb, 0)
    albSnow = jnp.where(isIce, phys.drySnowAlb, 0)

    # use albedo of wet surface if surface is thawing
    useWetAlb = (hIceActual > 0) & (TSurfLoc >= SurfMeltTemp)
    albIce = jnp.where(useWetAlb, phys.wetIceAlb, albIce)
    albSnow = jnp.where(useWetAlb, phys.wetSnowAlb, albSnow)

    # same for southern hermisphere
    south = (hIceActual > 0) & (vs.fCori < 0)
    albIce = jnp.where(south, phys.dryIceAlb_south, albIce)
    albSnow = jnp.where(south, phys.drySnowAlb_south, albSnow)
    useWetAlb_south = (hIceActual > 0) & (vs.fCori < 0) & (TSurfLoc >= SurfMeltTemp)
    albIce = jnp.where(useWetAlb_south, phys.wetIceAlb_south, albIce)
    albSnow = jnp.where(useWetAlb_south, phys.wetSnowAlb_south, albSnow)

    # if the snow thickness is smaller than hCut, use linear transition
    # between ice and snow albedo
    alb = jnp.where(isIce, albIce + hSnowActual / phys.hCut * (albSnow - albIce), 0)

    # if the snow thickness is larger than hCut, the snow is opaque for
    # shortwave radiation -> use snow albedo
    alb = jnp.where(hSnowActual > phys.hCut, albSnow, alb)

    # if no snow is present, use ice albedo
    alb = jnp.where(hSnowActual == 0, albIce, alb)

    ##### determine the shortwave radiative flux arriving at the     #####
    #####  ice-ocean interface after scattering through snow and ice #####

    # the fraction of shortwave radiative flux that arrives at the ocean
    # surface after passing the ice
    penetSWFrac = jnp.where(
        isIce, phys.shortwave * jnp.exp(-phys.iceShortwaveExtinction * hIceActual), 0
    )

    # if snow is present, all radiation is absorbed
    penetSWFrac = jnp.where(isSnow, 0, penetSWFrac)

    # shortwave radiative flux at the ocean-ice interface (+ = upward)
    IcePenetSW = jnp.where(isIce, -(1 - alb) * penetSWFrac * vs.SWdown, 0)

    # shortwave radiative flux convergence in the ice
    absorbedSW = jnp.where(isIce, (1 - alb) * (1 - penetSWFrac) * vs.SWdown, 0)

    # effective conductivity of the snow-ice system
    effConduct = jnp.where(
        isIce,
        phys.iceConduct
        * phys.snowConduct
        / (phys.snowConduct * hIceActual + phys.iceConduct * hSnowActual),
        0,
    )

    ##### calculate the heat fluxes #####

    def fluxes(t1: ArrayInput) -> tuple[Array, Array, Array, Array]:
        """Evaluate conductive/latent/net atmospheric flux and its derivative."""
        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2

        # saturation vapor pressure of snow/ice surface
        svp = 10 ** (-aa1 / t1 + aa2)

        # specific humidity at the surface
        q_s = jnp.where(isIce, bb1 * svp / (Ppascals - (1 - bb1) * svp), 0)

        # derivative of q_s w.r.t snow/ice surface temperature
        cc3t = 10 ** (aa1 / t1)
        dqs_dTs = jnp.where(isIce, cc1 * cc3t / ((cc2 - cc3t * Ppascals) ** 2 * t2), 0)

        # calculate the fluxes based on the surface temperature

        # conductive heat flux through ice and snow (+ = upward)
        F_c = jnp.where(isIce, effConduct * (TempFrz - t1), 0)

        # latent heat flux (sublimation) (+ = upward)
        F_lh = jnp.where(isIce, d1i * ug * (q_s - vs.aqh), 0)

        # long-wave surface heat flux (+ = upward)
        F_lwu = jnp.where(isIce, t4 * d3, 0)

        # sensible surface heat flux (+ = upward)
        F_sens = jnp.where(isIce, d1 * ug * (t1 - ATempLoc), 0)

        # upward seaice/snow surface heat flux to atmosphere
        F_ia = jnp.where(isIce, (-LWdownLoc - absorbedSW + F_lwu + F_sens + F_lh), 0)

        # derivative of F_ia w.r.t. snow/ice surf. temp
        dFia_dTs = jnp.where(isIce, 4 * d3 * t3 + d1 * ug + d1i * ug * dqs_dTs, 0)

        return F_c, F_lh, F_ia, dFia_dTs

    # iterate for the temperatue to converge (Newton-Raphson method)
    for _ in range(sett.surfaceTemperatureIterations):
        F_c, F_lh, F_ia, dFia_dTs = fluxes(TSurfLoc)

        # update surface temperature as solution of
        # F_c = F_ia + d/dT (F_c - F_ia) * delta T
        TSurfLoc = jnp.where(
            isIce, TSurfLoc + (F_c - F_ia) / (effConduct + dFia_dTs), 0
        )

        # add upper and lower boundary
        TSurfLoc = jnp.minimum(TSurfLoc, Tmelt)
        TSurfLoc = jnp.maximum(TSurfLoc, phys.celsius2K + sett.minTIce)

    # recalculate the fluxes based on the adjusted surface temperature
    F_c, F_lh, F_ia, dFia_dTs = fluxes(TSurfLoc)

    # set net ocean-ice flux and surface heat flux divergence based on
    # the direction of the conductive heat flux
    upCondFlux = F_c > 0
    F_io_net = jnp.where(upCondFlux, F_c, 0)
    F_ia_net = jnp.where(upCondFlux, 0, F_ia)

    # save updated surface temperature as output
    TSurfOut = jnp.where(isIce, TSurfLoc, TSurfIn)

    # freshwater flux due to sublimation [kg/m2] (+ = upward)
    FWsublim = jnp.where(isIce, F_lh / phys.lhSublim, 0)

    return TSurfOut, F_io_net, F_ia_net, IcePenetSW, FWsublim
