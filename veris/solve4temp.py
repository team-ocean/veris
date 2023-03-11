from veros.core.operators import numpy as npx
from veros import veros_kernel


@veros_kernel
def solve4temp(state, hIceActual, hSnowActual, TSurfIn, TempFrz):

    '''calculate heat fluxes through the ice and ice surface temperature'''

    vs = state.variables
    sett = state.settings


    ##### define local constants used for calculations #####

    # coefficients for the saturation vapor pressure equation
    aa1 = 2663.5
    aa2 = 12.537
    bb1 = 0.622
    bb2 = 1 - bb1
    Ppascals = 100000
    cc0 = 10**aa2
    cc1 = cc0 * aa1 * bb1 * Ppascals * npx.log(10)
    cc2 = cc0 * bb2
    
    # sensible heat constant
    d1 = sett.dalton * sett.cpAir * sett.rhoAir
    # latent heat constant
    d1i = sett.dalton * sett.lhSublim * sett.rhoAir

    # melting temperature of ice
    Tmelt = sett.celsius2K

    # temperature threshold for when to use wet albedo
    SurfMeltTemp = Tmelt + sett.wetAlbTemp

    # make local copies of downward longwave radiation, surface
    # and atmospheric temperatures
    TSurfLoc = TSurfIn
    LWdownLocCapped = npx.maximum(sett.minLWdown, vs.LWdown)
    ATempLoc = npx.maximum(sett.celsius2K + sett.minTAir, vs.ATemp)

    # set wind speed with lower boundary
    ug = npx.maximum(sett.wSpeedMin, vs.wSpeed)


    isIce = (hIceActual > 0)
    isSnow = (hSnowActual > 0)

    d3 = npx.where(isSnow, sett.snowEmiss, sett.iceEmiss) * sett.stefBoltz

    LWdownLoc = npx.where(isSnow, sett.snowEmiss, sett.iceEmiss) * LWdownLocCapped


    ##### determine albedo #####

    # use albedo of dry surface (if ice is present)
    albIce = npx.where(isIce, sett.dryIceAlb, 0)
    albSnow = npx.where(isIce, sett.drySnowAlb, 0)

    # use albedo of wet surface if surface is thawing
    useWetAlb = ((hIceActual > 0) & (TSurfLoc >= SurfMeltTemp))
    albIce = npx.where(useWetAlb, sett.wetIceAlb, albIce)
    albSnow = npx.where(useWetAlb, sett.wetSnowAlb, albSnow)

    # same for southern hermisphere
    south = ((hIceActual > 0) & (vs.fCori < 0))
    albIce = npx.where(south, sett.dryIceAlb_south, albIce)
    albSnow = npx.where(south, sett.drySnowAlb_south, albSnow)
    useWetAlb_south = ((hIceActual > 0) & (vs.fCori < 0) & (TSurfLoc >= SurfMeltTemp))
    albIce = npx.where(useWetAlb_south, sett.wetIceAlb_south, albIce)
    albSnow = npx.where(useWetAlb_south, sett.wetSnowAlb_south, albSnow)

    # if the snow thickness is smaller than hCut, use linear transition
    # between ice and snow albedo
    alb = npx.where(isIce, albIce + hSnowActual / sett.hCut * (albSnow - albIce), 0)

    # if the snow thickness is larger than hCut, the snow is opaque for
    # shortwave radiation -> use snow albedo
    alb = npx.where(hSnowActual > sett.hCut, albSnow, alb)

    # if no snow is present, use ice albedo
    alb = npx.where(hSnowActual == 0, albIce, alb)


    ##### determine the shortwave radiative flux arriving at the     #####
    #####  ice-ocean interface after scattering through snow and ice #####

    # the fraction of shortwave radiative flux that arrives at the ocean
    # surface after passing the ice
    penetSWFrac = npx.where(isIce, sett.shortwave * npx.exp(-1.5 * hIceActual), 0)

    # if snow is present, all radiation is absorbed
    penetSWFrac = npx.where(isSnow, 0, penetSWFrac)

    # shortwave radiative flux at the ocean-ice interface (+ = upward)
    IcePenetSW = npx.where(isIce, -(1 - alb) * penetSWFrac * vs.SWdown, 0)

    # shortwave radiative flux convergence in the ice
    absorbedSW = npx.where(isIce, (1 - alb) * (1 - penetSWFrac) * vs.SWdown, 0)
    
    # effective conductivity of the snow-ice system
    effConduct = npx.where(isIce, sett.iceConduct * sett.snowConduct / (
                    sett.snowConduct * hIceActual + sett.iceConduct * hSnowActual), 0)


    ##### calculate the heat fluxes #####

    def fluxes(t1):

        t2 = t1 * t1
        t3 = t2 * t1
        t4 = t2 * t2


        # saturation vapor pressure of snow/ice surface
        svp = 10**(- aa1 / t1 + aa2)

        # specific humidity at the surface
        q_s = npx.where(isIce, bb1 * svp / (Ppascals - (1 - bb1) * svp), 0)

        # derivative of q_s w.r.t snow/ice surface temperature
        cc3t = 10**(aa1 / t1)
        dqs_dTs = npx.where(isIce, cc1 * cc3t / ((cc2 - cc3t * Ppascals)**2 * t2), 0)

        # calculate the fluxes based on the surface temperature

        # conductive heat flux through ice and snow (+ = upward)
        F_c  = npx.where(isIce, effConduct * (TempFrz - t1), 0)

        # latent heat flux (sublimation) (+ = upward)
        F_lh = npx.where(isIce, d1i * ug * (q_s - vs.aqh), 0)

        # long-wave surface heat flux (+ = upward)
        F_lwu = npx.where(isIce, t4 * d3, 0)

        # sensible surface heat flux (+ = upward)
        F_sens = npx.where(isIce, d1 * ug * (t1 - ATempLoc), 0)

        # upward seaice/snow surface heat flux to atmosphere
        F_ia = npx.where(isIce, (- LWdownLoc - absorbedSW + F_lwu
                                + F_sens + F_lh), 0)

        # derivative of F_ia w.r.t. snow/ice surf. temp
        dFia_dTs = npx.where(isIce, 4 * d3 * t3 + d1 * ug
                                    + d1i * ug * dqs_dTs, 0)

        return F_c, F_lh, F_ia, dFia_dTs

    # iterate for the temperatue to converge (Newton-Raphson method)
    for i in range(6):

        F_c, F_lh, F_ia, dFia_dTs = fluxes(TSurfLoc)

        # update surface temperature as solution of
        # F_c = F_ia + d/dT (F_c - F_ia) * delta T
        TSurfLoc = npx.where(isIce, TSurfLoc + (F_c - F_ia)
                                    / (effConduct + dFia_dTs), 0)

        # add upper and lower boundary
        TSurfLoc = npx.minimum(TSurfLoc, Tmelt)
        TSurfLoc = npx.maximum(TSurfLoc, sett.celsius2K + sett.minTIce)

    # recalculate the fluxes based on the adjusted surface temperature
    F_c, F_lh, F_ia, dFia_dTs = fluxes(TSurfLoc)

    # set net ocean-ice flux and surface heat flux divergence based on
    # the direction of the conductive heat flux
    upCondFlux = (F_c > 0)
    F_io_net = npx.where(upCondFlux, F_c, 0)
    F_ia_net = npx.where(upCondFlux, 0, F_ia)

    # save updated surface temperature as output
    TSurfOut = npx.where(isIce, TSurfLoc, TSurfIn)

    # freshwater flux due to sublimation [kg/m2] (+ = upward)
    FWsublim = npx.where(isIce, F_lh / sett.lhSublim, 0)

    return TSurfOut, F_io_net, F_ia_net, F_ia, IcePenetSW, FWsublim