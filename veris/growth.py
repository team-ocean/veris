from veros.core.operators import numpy as npx
from veros.core.operators import update, at
from veros import veros_kernel

from veris.solve4temp import solve4temp


@veros_kernel
def Growth(state):

    '''calculate thermodynamic change of ice and snow thickness and ice cover fraction
    due to atmospheric and ocean surface forcing'''

    vs = state.variables
    sett = state.settings


    ##### initializations #####

    # F_ia_net: net heat flux divergence at the sea ice/snow surface [W/m2]
    # = 0: surface heat loss is balanced by upward conductive fluxes
    # < 0: net heat flux convergence at the ice/ snow surface

    # F_io_net: net upward conductive heat flux through sea ice and snow [W/m2]

    # initialize three dimensional arrays accounting for the thickness categories of the ice
    # (using * 1 ensures that a new array is created for each variable. otherwise they would
    # all point to the same one)
    ones3d = npx.zeros((*vs.iceMask.shape,sett.nITC))
    hIceActual_mult = ones3d * 1
    hSnowActual_mult = ones3d * 1
    F_io_net_mult = ones3d * 1
    F_ia_net_mult = ones3d * 1
    IcePenetSW_mult = ones3d * 1
    FWsublim_mult = ones3d * 1

    # constants for converting heat fluxes into growth rates
    qi = 1 / (sett.rhoIce * sett.lhFusion)
    qs = 1 / (sett.rhoSnow * sett.lhFusion)

    # store sea ice fields prior to any thermodynamical changes
    noIce = ((vs.hIceMean == 0) | (vs.Area == 0))
    hIceMeanpreTH = vs.hIceMean * ~noIce
    hSnowMeanpreTH = vs.hSnowMean * ~noIce
    AreapreTH = vs.Area * ~noIce

    # compute actual ice and snow thickness using the regularized area.
    # ice or snow thickness divided by Area does not work if Area -> 0,
    # therefore the regularization
    isIce = (hIceMeanpreTH > 0)
    regArea =  npx.sqrt(AreapreTH**2 + sett.Area_reg)
    recip_regArea = 1 / regArea

    hIceActual = npx.where(isIce, hIceMeanpreTH * recip_regArea, 0)
    recip_hIceActual = AreapreTH / npx.sqrt(hIceMeanpreTH**2 + sett.hIce_reg)
    hSnowActual = npx.where(isIce, hSnowMeanpreTH * recip_regArea, 0)

    hIceActual = npx.maximum(hIceActual, 0.05)


    ##### evaluate precipitation as snow or rain #####

    # if there is ice and the temperature is below the freezing point,
    # the precipitation falls and accumulates as snow
    tmp = ((AreapreTH > 0) & (npx.mean(TIce_mult,axis=2) < sett.celsius2K))

    # snow accumulation rate over ice [m/s]
    # the snowfall is given in water equivalent, therefore it also needs to be muliplied with rhoFresh2rhoSnow
    SnowAccRateOverIce = vs.snowfall
    SnowAccRateOverIce = npx.where(tmp, SnowAccRateOverIce + vs.precip,
                                SnowAccRateOverIce) * sett.rhoFresh2rhoSnow

    # the precipitation rate over the ice which goes immediately into the
    # ocean (flowing through cracks in the ice). if the temperature is
    # above the freezing point, the precipitation remains wet and runs
    # into the ocean
    PrecipRateOverIceSurfaceToSea = npx.where(tmp, 0, vs.precip)

    # total snow accumulation over ice [m]
    SnowAccOverIce = SnowAccRateOverIce * AreapreTH * sett.deltatTherm


    ##### calculate heat fluxes through the ice #####

    # set ice and snow thickness categories to account for thicknes variations in one grid cell
    TIce_mult = ones3d * 1

    for l in range(0, sett.nITC):
        # the ice categories all have the same initial temperature
        TIce_mult = update(TIce_mult, at[:,:,l], vs.TSurf)

        # set relative thickness of ice and snow categories
        pFac = (2 * (l + 1) - 1) * sett.recip_nITC

        # actual snow and ice thickness within each category
        hIceActual_mult = update(hIceActual_mult, at[:,:,l], hIceActual * pFac)
        hSnowActual_mult = update(hSnowActual_mult, at[:,:,l], hSnowActual * pFac)

    # freezing temperature
    TempFrz = sett.tempFrz + sett.dtempFrz_dS * vs.ocSalt + sett.celsius2K

    # calculate heat fluxes
    for l in range(sett.nITC):
        output = solve4temp(state, hIceActual_mult[:,:,l], hSnowActual_mult[:,:,l],
            TIce_mult[:,:,l], TempFrz)

        TIce_mult = update(TIce_mult, at[:,:,l], output[0])
        F_io_net_mult = update(F_io_net_mult, at[:,:,l], output[1])
        F_ia_net_mult = update(F_ia_net_mult, at[:,:,l], output[2])
        IcePenetSW_mult = update(IcePenetSW_mult, at[:,:,l], output[4])
        FWsublim_mult = update(FWsublim_mult, at[:,:,l], output[5])

    # update surface temperature and fluxes
    TSurf = npx.sum(TIce_mult, axis=2) * sett.recip_nITC

    # multplying the fluxes with the area changes them from mean fluxes
    # for the ice part of the cell to mean fluxes for the whole cell
    F_io_net = npx.sum(F_io_net_mult, axis=2) * sett.recip_nITC * AreapreTH
    F_ia_net = npx.sum(F_ia_net_mult, axis=2) * sett.recip_nITC * AreapreTH
    IcePenetSW = npx.sum(IcePenetSW_mult, axis=2) * sett.recip_nITC * AreapreTH
    # FWsublim = npx.sum(FWsublim_mult, axis=2) * sett.recip_nITC * AreapreTH


    ##### calculate growth rates of ice and snow #####

    # the ice growth rate beneath ice is given by the upward conductive
    # flux F_io_net and qi:
    IceGrowthRateUnderExistingIce = F_io_net * qi
    IceGrowthRateUnderExistingIce = npx.where(AreapreTH == 0,
                                        0, IceGrowthRateUnderExistingIce)

    # the potential snow melt rate if all snow surface heat flux
    # convergence (F_ia_net < 0) goes to melting snow [m/s]
    PotSnowMeltRateFromSurf = - F_ia_net * qs

    # the thickness of snow that can be melted in one time step:
    PotSnowMeltFromSurf = PotSnowMeltRateFromSurf * sett.deltatTherm

    # if the heat flux convergence could melt more snow than is actually
    # there, the excess is used to melt ice

    # case 1: snow will remain after melting, i.e. all of the heat flux
    # convergence will be used up to melt snow
    # case 2: all snow will be melted if the potential snow melt
    # height is larger or equal to the actual snow height. if there is
    # an excess of heat flux convergence after snow melting, it will
    # be used to melt ice

    # (use hSnowActual for comparison with the MITgcm)
    allSnowMelted = (PotSnowMeltFromSurf >= hSnowMeanpreTH)

    # the actual thickness of snow to be melted by snow surface
    # heat flux convergence [m]
    SnowMeltFromSurface = npx.where(allSnowMelted, hSnowMeanpreTH,
                                                    PotSnowMeltFromSurf)

    # the actual snow melt rate due to snow surface heat flux convergence [m/s]
    SnowMeltRateFromSurface = npx.where(allSnowMelted,
                SnowMeltFromSurface * sett.recip_deltatTherm,
                PotSnowMeltRateFromSurf)

    # the actual surface heat flux convergence used to melt snow [W/m2]
    SurfHeatFluxConvergToSnowMelt = npx.where(allSnowMelted,
                - hSnowMeanpreTH * sett.recip_deltatTherm / qs, F_ia_net)

    # the surface heat flux convergence is reduced by the amount that
    # is used for melting snow:
    F_ia_net = F_ia_net - SurfHeatFluxConvergToSnowMelt

    # the remaining heat flux convergence is used to melt ice:
    IceGrowthRateFromSurface = F_ia_net * qi

    # the total ice growth rate is then:
    # (remove * recip_regArea for comparison with the MITgcm)
    NetExistingIceGrowthRate = (IceGrowthRateUnderExistingIce
                                + IceGrowthRateFromSurface) * recip_regArea


    ##### calculate the ice melting due to mixed layer temperature #####

    tmpscal0 = 0.4
    tmpscal1 = 7 / tmpscal0
    tmpscal2 = sett.stantonNr * sett.uStarBase * sett.rhoSea * sett.cpWater

    # the ocean temperature cannot be lower than the freezing temperature
    surf_theta = npx.maximum(vs.theta, TempFrz)

    # mltf = mixed layer turbulence factor (determines how much of the temperature
    # difference is used for heat flux)
    mltf = 1 + (sett.McPheeTaperFac - 1) / (1 + npx.exp((AreapreTH - tmpscal0) * tmpscal1))

    # heat flux from ocean to the ice (+ = upward)
    F_oi = - tmpscal2 * (surf_theta - TempFrz) * mltf
    IceGrowthRateMixedLayer = F_oi * qi


    ##### calculate the ice growth in open water due to heat loss to atmosphere #####

    # in this configuration, all of the incoming shortwave radiation is absorbed in the
    # uppermost ocean cell. If this is not the case, the shortwave radiation that passes
    # the ocean surface cell is to be distracted from Qnet as it is not part of the heat
    # flux convergence
    # define swFracPass as the fraction of the shortwave radiation that passes the ocean
    # surface cell
    # Qsw_in_first_layer = vs.Qsw * ( 1 - swFracPass )
    # IceGrowthRateOpenWater = qi * ( vs.Qnet - vs.Qsw + Qsw_in_first_layer )
    # using swFracPass = 0 yields the same result as just using 
    # IceGrowthRateOpenWater = qi * vs.Qnet

    IceGrowthRateOpenWater = qi * vs.Qnet


    ##### calculate change in ice, snow thicknesses and area #####

    # calculate thickness derivatives of ice and snow
    dhIceMean_dt = NetExistingIceGrowthRate * AreapreTH + \
        IceGrowthRateOpenWater * (1 - AreapreTH) + IceGrowthRateMixedLayer
    dhSnowMean_dt = (SnowAccRateOverIce - SnowMeltRateFromSurface) * AreapreTH

    tmpscal0 =  0.5 * recip_hIceActual

    # ice growth open water (due to ocean-atmosphere fluxes)
    # reduce ice cover if the open water growth rate is negative
    dArea_oaFlux = tmpscal0 * IceGrowthRateOpenWater * (1 - AreapreTH)

    # increase ice cover if the open water growth rate is positive
    tmp = ((IceGrowthRateOpenWater > 0) & (
            (AreapreTH > 0) | (dhIceMean_dt > 0)))
    dArea_oaFlux = npx.where(tmp, IceGrowthRateOpenWater
                             * (1 - AreapreTH), dArea_oaFlux)

    # multiply with lead closing factor
    dArea_oaFlux = npx.where((tmp & (vs.fCori < 0)),
                                dArea_oaFlux * sett.recip_h0_south, 
                                dArea_oaFlux * sett.recip_h0)

    # ice growth mixed layer (due to ocean-ice fluxes)
    # (if the ocean is warmer than the ice: IceGrowthRateMixedLayer > 0.
    # the supercooled state of the ocean is ignored/ does not lead to ice
    # growth. ice growth is only due to fluxes calculated by solve4temp)
    dArea_oiFlux = npx.where(IceGrowthRateMixedLayer <= 0,
        tmpscal0 * IceGrowthRateMixedLayer, 0)

    # ice growth over ice (due to ice-atmosphere fluxes)
    # (NetExistingIceGrowthRate leads to vertical and lateral melting but
    # only to vertical growing. lateral growing is covered by
    # IceGrowthRateOpenWater)
    dArea_iaFlux = npx.where((NetExistingIceGrowthRate <= 0) & (hIceMeanpreTH > 0),
        tmpscal0 * NetExistingIceGrowthRate * AreapreTH, 0)

    # total change in area
    dArea_dt = dArea_oaFlux + dArea_oiFlux + dArea_iaFlux


    ######  update ice, snow thickness and area #####

    Area = AreapreTH + dArea_dt * vs.iceMask * sett.deltatTherm
    hIceMean = hIceMeanpreTH + dhIceMean_dt * vs.iceMask * sett.deltatTherm
    hSnowMean = hSnowMeanpreTH + dhSnowMean_dt * vs.iceMask * sett.deltatTherm

    # set boundaries:
    Area = npx.clip(Area, 0, 1)
    hIceMean = npx.maximum(hIceMean, 0)
    hSnowMean = npx.maximum(hSnowMean, 0)

    noIce = ((hIceMean == 0) | (Area == 0))
    Area *= ~noIce
    hSnowMean *= ~noIce
    hIceMean *= ~noIce

    # change of ice thickness due to conversion of snow to ice if snow
    # is submerged with water
    h_sub = (hSnowMean * sett.rhoSnow + hIceMean * sett.rhoIce) * sett.recip_rhoSea
    d_hIceMeanByFlood = npx.maximum(0, h_sub - hIceMean)
    hIceMean = hIceMean + d_hIceMeanByFlood
    hSnowMean = hSnowMean - d_hIceMeanByFlood * sett.rhoIce2rhoSnow

    recip_hIceMean = 1 / npx.sqrt(hIceMean**2 + sett.hIce_reg)


    ##### calculate output to ocean #####

    # effective shortwave heating rate
    Qsw = IcePenetSW * AreapreTH + vs.Qsw * (1 - AreapreTH)

    # the actual ice volume change over the time step [m3/m2]
    hIceMeanChange = hIceMean - hIceMeanpreTH

    # the net melted snow thickness [m3/m2] (positive if the snow
    # thickness decreases/ melting occurs)
    MeltedSnowMean = hSnowMeanpreTH + SnowAccOverIce - hSnowMean

    # the energy required to melt or form the new ice volume [J/m2]
    EnergyForIceChange = hIceMeanChange / qi

    # the net energy flux out of the ocean [J/m2]
    NetEnergyFluxOutOfOcean = (AreapreTH * (F_ia_net + F_io_net + IcePenetSW)
                + (1 - AreapreTH) * vs.Qnet) * sett.deltatTherm

    # energy taken out of the ocean which is not used for sea ice growth [J].
    # If the net energy flux out of the ocean is balanced by the latent
    # heat of fusion, the temperature of the mixed layer will not change
    ResidualEnergyOutOfOcean = NetEnergyFluxOutOfOcean - EnergyForIceChange

    # total heat flux out of the ocean [W/m2]
    Qnet = ResidualEnergyOutOfOcean * sett.recip_deltatTherm

    # the freshwater contribution to (from) the ocean due to melting (growing)
    # of ice [m3/m2] (positive for melting)
    # in the case of non-zero ice salinity, the freshwater contribution
    # is reduced by the salinity ratio of ice to water.
    # if the liquid cell has a lower salinity than the specified salinity
    # of sea ice, then assume the sea ice is completely fresh
    # (if the water is fresh, no salty sea ice can form)
    FreshwaterContribFromIce = - hIceMeanChange * sett.rhoIce2rhoFresh * \
                npx.where(((vs.ocSalt > 0) & (vs.ocSalt > sett.saltIce_ref)),
                            (1 - sett.saltIce_ref/vs.ocSalt), 1)

    # the variables os_hIceMean and os_hSnowMean are negative ice and snow thicknesses
    # that were calculated in the advection routine. the thicknesses were cut off at zero but
    # to ensure mass conservation, the overshoots are treated as melted

    # salt flux into the ocean due to ice formation [kg/s m2]
    # this is an actual salt flux 
    tmpscal0 = npx.minimum(sett.saltIce_ref, vs.ocSalt)
    saltflux = (hIceMeanChange + vs.os_hIceMean) * tmpscal0 \
        * vs.iceMask * sett.rhoIce * sett.recip_deltatTherm

    # the freshwater contribution to the ocean from melting snow [m]
    FreshwaterContribFromSnowMelt = MeltedSnowMean / sett.rhoFresh2rhoSnow

    # evaporation minus precipitation minus runoff (salt flux into ocean) [kg/m2 s]
    # this is a virtual salt flux (actually just a negative freshwater flux, it still needs to be
    # weighted with the ocean surface salinity). if there is a flux of freshwater into the ocean,
    # the salinity has to decrease. as the ocean model treats the water volume as constant, the
    # adding of freshwater is therefore treated as salt flux out of the ocean.
    # (the name EmPmR is convention and assumes a positive sign of evap for evaporation. in this
    # configuration, evap is a freshwater flux, i.e. <0 increases salinity, therefore the additional
    # minus in front of evap)
    EmPmR = vs.iceMask *  ((- vs.evap - vs.precip) * (1 - AreapreTH) \
        - PrecipRateOverIceSurfaceToSea * AreapreTH - vs.runoff - (
        FreshwaterContribFromIce + FreshwaterContribFromSnowMelt) \
            / sett.deltatTherm) * sett.rhoFresh + vs.iceMask * (
        vs.os_hIceMean * sett.rhoIce + vs.os_hSnowMean * sett.rhoSnow) \
            * sett.recip_deltatTherm

    # convert freshwater flux to salt flux and combine virtual and actual salt flux
    forc_salt_surface = EmPmR * vs.ocSalt / sett.rhoFresh + saltflux

    # sea ice + snow load on the sea surface
    SeaIceLoad = hIceMean * sett.rhoIce + hSnowMean * sett.rhoSnow

    return hIceMean, hSnowMean, Area, TSurf, EmPmR, forc_salt_surface, Qsw, Qnet, \
        SeaIceLoad, IcePenetSW, recip_hIceMean