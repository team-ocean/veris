from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel

from veris.fill_overlap import fill_overlap

# in this routine, the thermodynamic time step is used instead of the dynamic one.
# this has historical reasons as with lower resolutions, the dynamics change much
# slower than the thermodynamics (thermodynamics have a daily cycle). calculating
# the ice velocity as often as the thermodynamics was unnecessarily expensive but
# the advection is still done with the faster thermodynamic timestep as the ice
# thickness changes inbetween dynamics timesteps.


@veros_kernel
def Advection(state):
    """retrieve changes in sea ice fields"""

    vs = state.variables

    hIceMean = calc_Advection(state, vs.hIceMean)
    hSnowMean = calc_Advection(state, vs.hSnowMean)
    Area = calc_Advection(state, vs.Area)

    return hIceMean, hSnowMean, Area


@veros_kernel
def calc_Advection(state, field):
    """calculate change in sea ice field due to advection"""

    vs = state.variables
    sett = state.settings

    # retrieve cell faces
    xA = vs.dyG * vs.iceMaskU
    yA = vs.dxG * vs.iceMaskV

    # calculate ice transport
    uTrans = vs.uIce * xA
    vTrans = vs.vIce * yA

    # make local copy of field prior to advective changes
    fieldLoc = field

    # calculate zonal advective fluxes
    ZonalFlux = calc_ZonalFlux(state, fieldLoc, uTrans)

    # update field according to zonal fluxes
    if sett.extensiveFld:
        fieldLoc = fieldLoc - sett.deltatTherm * vs.maskInC * vs.recip_rA * (
            npx.roll(ZonalFlux, -1, 0) - ZonalFlux
        )
    else:
        fieldLoc = (
            fieldLoc
            - sett.deltatTherm
            * vs.maskInC
            * vs.recip_rA
            * vs.recip_hIceMean
            * (
                (npx.roll(ZonalFlux, -1, 1) - ZonalFlux)
                - (npx.roll(vs.uTrans, -1, 0) - vs.uTrans) * field
            )
        )

    # calculate meridional advective fluxes
    MeridionalFlux = calc_MeridionalFlux(state, fieldLoc, vTrans)

    # update field according to meridional fluxes
    if sett.extensiveFld:
        fieldLoc = fieldLoc - sett.deltatTherm * vs.maskInC * vs.recip_rA * (
            npx.roll(MeridionalFlux, -1, 1) - MeridionalFlux
        )
    else:
        fieldLoc = (
            fieldLoc
            - sett.deltatTherm
            * vs.maskInC
            * vs.recip_rA
            * vs.recip_hIceMean
            * (
                (npx.roll(MeridionalFlux, -1, 0) - MeridionalFlux)
                - (npx.roll(vs.vTrans, -1, 1) - vs.vTrans) * field
            )
        )

    # apply mask
    fieldLoc = fieldLoc * vs.iceMask

    return fieldLoc


@veros_kernel
def calc_ZonalFlux(state, field, uTrans):
    """calculate the zonal advective flux using the second order flux limiter method"""

    vs = state.variables
    sett = state.settings

    maskLocW = vs.iceMaskU * vs.maskInU

    # CFL number of zonal flow
    uCFL = npx.abs(vs.uIce * sett.deltatTherm * vs.recip_dxC)

    # calculate slope ratio Cr
    Rjp = (field[3:, :] - field[2:-1, :]) * maskLocW[3:, :]
    Rj = (field[2:-1, :] - field[1:-2, :]) * maskLocW[2:-1, :]
    Rjm = (field[1:-2, :] - field[:-3, :]) * maskLocW[1:-2, :]

    Cr = npx.where(uTrans[2:-1, :] > 0, Rjm, Rjp)
    Cr = npx.where(
        npx.abs(Rj) * sett.CrMax > npx.abs(Cr),
        Cr / Rj,
        npx.sign(Cr) * sett.CrMax * npx.sign(Rj),
    )
    Cr = limiter(Cr)

    # zonal advective flux for the given field
    ZonalFlux = npx.zeros_like(vs.iceMask)
    ZonalFlux = update(
        ZonalFlux,
        at[2:-1, :],
        uTrans[2:-1, :] * (field[2:-1, :] + field[1:-2, :]) * 0.5
        - npx.abs(uTrans[2:-1, :]) * ((1 - Cr) + uCFL[2:-1, :] * Cr) * Rj * 0.5,
    )
    ZonalFlux = fill_overlap(state, ZonalFlux)

    return ZonalFlux


@veros_kernel
def calc_MeridionalFlux(state, field, vTrans):
    """calculate the meridional advective flux using the second order flux limiter method"""

    vs = state.variables
    sett = state.settings

    maskLocS = vs.iceMaskV * vs.maskInV

    # CFL number of meridional flow
    vCFL = npx.abs(vs.vIce * sett.deltatTherm * vs.recip_dyC)

    # calculate slope ratio Cr
    Rjp = (field[:, 3:] - field[:, 2:-1]) * maskLocS[:, 3:]
    Rj = (field[:, 2:-1] - field[:, 1:-2]) * maskLocS[:, 2:-1]
    Rjm = (field[:, 1:-2] - field[:, :-3]) * maskLocS[:, 1:-2]

    Cr = npx.where(vTrans[:, 2:-1] > 0, Rjm, Rjp)
    Cr = npx.where(
        npx.abs(Rj) * sett.CrMax > npx.abs(Cr),
        Cr / Rj,
        npx.sign(Cr) * sett.CrMax * npx.sign(Rj),
    )
    Cr = limiter(Cr)

    # meridional advective flux for the given field
    MeridionalFlux = npx.zeros_like(vs.iceMask)
    MeridionalFlux = update(
        MeridionalFlux,
        at[:, 2:-1],
        vTrans[:, 2:-1] * (field[:, 2:-1] + field[:, 1:-2]) * 0.5
        - npx.abs(vTrans[:, 2:-1]) * ((1 - Cr) + vCFL[:, 2:-1] * Cr) * Rj * 0.5,
    )
    MeridionalFlux = fill_overlap(state, MeridionalFlux)

    return MeridionalFlux


@veros_kernel
def limiter(Cr):
    # return 0       (upwind)
    # return 1       (Lax-Wendroff)
    # return np.max((0, np.min((1, Cr))))    (Min-Mod)
    return npx.maximum(0, npx.maximum(npx.minimum(1, 2 * Cr), npx.minimum(2, Cr)))