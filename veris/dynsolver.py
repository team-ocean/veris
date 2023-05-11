from veros.core.operators import numpy as npx
from veros import veros_kernel
from veris.freedrift_solver import freedrift_solver
from veris.evp_solver import evp_solver


@veros_kernel
def tauXY(state):
    """calculate surface stress from wind and ice velocities"""

    vs = state.variables
    sett = state.settings

    sinWin = npx.sin(npx.deg2rad(sett.airTurnAngle))
    cosWin = npx.cos(npx.deg2rad(sett.airTurnAngle))

    # calculate relative wind at c-points
    urel = vs.uWind - 0.5 * (vs.uIce + npx.roll(vs.uIce, -1, 0))
    vrel = vs.vWind - 0.5 * (vs.vIce + npx.roll(vs.vIce, -1, 1))

    # calculate wind speed and set lower boundary
    windSpeed_sq = urel**2 + vrel**2
    windSpeed = npx.where(
        windSpeed_sq < sett.wSpeedMin**2, sett.wSpeedMin, npx.sqrt(windSpeed_sq)
    )

    # calculate air-ice drag coefficient
    CDAir = (
        npx.where(vs.fCori < 0, sett.airIceDrag_south, sett.airIceDrag)
        * sett.rhoAir
        * windSpeed
    )

    # calculate surface stress
    tauX = CDAir * (cosWin * urel - npx.sign(vs.fCori) * sinWin * vrel)
    tauY = CDAir * (cosWin * vrel + npx.sign(vs.fCori) * sinWin * urel)

    # interpolate to u- and v-points
    tauX = 0.5 * (tauX + npx.roll(tauX, 1, 0)) * vs.iceMaskU
    tauY = 0.5 * (tauY + npx.roll(tauY, 1, 1)) * vs.iceMaskV

    return tauX, tauY


@veros_kernel
def WindForcingXY(state):
    """calculate surface forcing due to wind and ocean surface tilt"""

    vs = state.variables
    sett = state.settings

    # calculate surface stresses from wind and ice velocities
    tauX, tauY = tauXY(state)

    # calculate forcing by surface stress
    WindForcingX = tauX * vs.AreaW
    WindForcingY = tauY * vs.AreaS

    # calculate geopotential anomaly. the surface pressure and sea ice load are
    # used as they affect the sea surface height anomaly
    phiSurf = sett.gravity * vs.ssh_an
    if state.settings.useRealFreshWaterFlux:
        phiSurf = (
            phiSurf
            + (vs.surfPress + vs.SeaIceLoad * sett.gravity * sett.seaIceLoadFac)
            * sett.recip_rhoSea
        )
    else:
        phiSurf = phiSurf + vs.surfPress * sett.recip_rhoSea

    # add in tilt
    WindForcingX = WindForcingX - vs.SeaIceMassU * vs.recip_dxC * (
        phiSurf - npx.roll(phiSurf, 1, 0)
    )
    WindForcingY = WindForcingY - vs.SeaIceMassV * vs.recip_dyC * (
        phiSurf - npx.roll(phiSurf, 1, 1)
    )

    return WindForcingX, WindForcingY


@veros_kernel
def IceVelocities(state):
    """calculate ice velocities from surface and ocean forcing"""

    sett = state.settings

    if sett.useFreedrift:
        uIce, vIce = freedrift_solver(state)

    if sett.useEVP:
        uIce, vIce = evp_solver(state)

    return uIce, vIce
