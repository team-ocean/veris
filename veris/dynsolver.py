from functools import partial

import jax
import jax.numpy as jnp

from veris.evp_solver import evp_solver
from veris.freedrift_solver import freedrift_solver


@partial(jax.jit, static_argnames=["sett"])
def tauXY(vs, sett):
    """calculate surface stress from wind and ice velocities"""

    sinWin = jnp.sin(jnp.deg2rad(sett.airTurnAngle))
    cosWin = jnp.cos(jnp.deg2rad(sett.airTurnAngle))

    if sett.useRelativeWind:
        # calculate relative wind at c-points
        urel = vs.uWind - 0.5 * (vs.uIce + jnp.roll(vs.uIce, -1, 0))
        vrel = vs.vWind - 0.5 * (vs.vIce + jnp.roll(vs.vIce, -1, 1))
    else:
        # only use wind for the wind stress calculation
        urel = vs.uWind
        vrel = vs.vWind

    # calculate wind speed and set lower boundary
    windSpeed_sq = urel**2 + vrel**2
    windSpeed = jnp.where(
        windSpeed_sq < sett.wSpeedMin**2, sett.wSpeedMin, jnp.sqrt(windSpeed_sq)
    )

    # calculate air-ice drag coefficient
    CDAir = (
        jnp.where(vs.fCori < 0, sett.airIceDrag_south, sett.airIceDrag)
        * sett.rhoAir
        * windSpeed
    )

    # calculate surface stress
    tauX = CDAir * (cosWin * urel - jnp.sign(vs.fCori) * sinWin * vrel)
    tauY = CDAir * (cosWin * vrel + jnp.sign(vs.fCori) * sinWin * urel)

    # interpolate to u- and v-points
    tauX = 0.5 * (tauX + jnp.roll(tauX, 1, 0)) * vs.iceMaskU
    tauY = 0.5 * (tauY + jnp.roll(tauY, 1, 1)) * vs.iceMaskV

    return tauX, tauY


@partial(jax.jit, static_argnames=["sett"])
def WindForcingXY(vs, sett):
    """calculate surface forcing due to wind and ocean surface tilt"""

    # calculate surface stresses from wind and ice velocities
    tauX, tauY = tauXY(vs, sett)

    # calculate forcing by surface stress
    WindForcingX = tauX * vs.AreaW
    WindForcingY = tauY * vs.AreaS

    # calculate geopotential anomaly. the surface pressure and sea ice load are
    # used as they affect the sea surface height anomaly
    phiSurf = sett.gravity * vs.ssh_an
    if sett.useRealFreshWaterFlux:
        phiSurf = (
            phiSurf
            + (vs.surfPress + vs.SeaIceLoad * sett.gravity * sett.seaIceLoadFac)
            * sett.recip_rhoSea
        )
    else:
        phiSurf = phiSurf + vs.surfPress * sett.recip_rhoSea

    # add in tilt
    WindForcingX = WindForcingX - vs.SeaIceMassU * vs.recip_dxC * (
        phiSurf - jnp.roll(phiSurf, 1, 0)
    )
    WindForcingY = WindForcingY - vs.SeaIceMassV * vs.recip_dyC * (
        phiSurf - jnp.roll(phiSurf, 1, 1)
    )

    return WindForcingX, WindForcingY


@partial(jax.jit, static_argnames=["sett", "axis_names"])
def IceVelocities(vs, sett, *, axis_names: tuple[str, ...] = ()):
    """Calculate ice velocities, reducing EVP diagnostics over supplied mesh axes."""

    if sett.useFreedrift:
        uIce, vIce = freedrift_solver(vs, sett)
        sigma1 = vs.sigma1
        sigma2 = vs.sigma2
        sigma12 = vs.sigma12

    if sett.useEVP:
        uIce, vIce, sigma1, sigma2, sigma12 = evp_solver(
            vs, sett, axis_names=axis_names
        )

    return uIce, vIce, sigma1, sigma2, sigma12
