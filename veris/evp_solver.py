from veros.core.operators import numpy as npx
from veros.core.operators import update, at, for_loop
from veros import veros_kernel

from veris.averaging import c_point_to_z_point
from veris.dynamics_routines import (
    strainrates,
    viscosities,
    ocean_drag_coeffs,
    basal_drag_coeffs,
    stressdiv,
    stress,
    side_drag,
)
from veris.global_sum import global_sum
from veris.fill_overlap import fill_overlap_uv

printEvpResidual = False
plotEvpResidual = False

@veros_kernel
def evp_solver_body(iEVP, arg_body):
    """loop body of the elastic-viscous-plastic solver
    the components of the strain rate tensor and stress tensor are calculated
    following Hibler (1979). the time stepping of the stress and velocity
    equations is done following Kimmritz (2016)
    """

    (
        state,
        uIce,
        vIce,
        uIceNm1,
        vIceNm1,
        sigma1,
        sigma2,
        sigma12,
        denom1,
        denom2,
        EVPcFac,
        evpAlphaC,
        evpAlphaZ,
        evpBetaU,
        evpBetaV,
        resSig,
        resU,
    ) = arg_body

    vs = state.variables
    sett = state.settings

    if sett.computeEvpResidual:
        # save previous (p-1) iteration for residual computation
        sig11Pm1 = sigma11
        sig22Pm1 = sigma22
        sig12Pm1 = sigma12
        uIcePm1 = uIce
        vIcePm1 = vIce
    
    e11, e22, e12 = strainrates(state, uIce, vIce)
    zeta, eta, press = viscosities(state, e11, e22, e12)
    #sig11, sig22, sig12 = stress(state, e11, e22, e12, zeta, eta, press)

    # calculate adaptive relaxation parameters
    if sett.useAdaptiveEVP:
        evpAlphaC = (
            npx.sqrt(zeta * EVPcFac / npx.maximum(vs.SeaIceMassC, 1e-4) * vs.recip_rA)
            * vs.iceMask
        )
        
        evpAlphaC = npx.maximum(evpAlphaC, sett.aEVPalphaMin)
        denom1 = 1.0 / evpAlphaC
        denom2 = denom1

    # copied from the MITgcm
    evpRevFac = 1
    recip_evpRevFac = 0.25

    # define principle strain rate components
    ep = e11 + e22
    em = e11 - e22

    # used to calculate the components of the stress tensor
    divergence = 2 * zeta * ep - press
    tension    = 2 * zeta * em
    shear      = 2 * c_point_to_z_point(state, zeta) * e12 

    # step principal stress components
    sigma1 = ( sigma1 * ( evpAlphaC - evpRevFac ) + divergence ) * denom1 * vs.iceMask
    sigma2 = ( sigma2 * ( evpAlphaC - evpRevFac ) + tension * recip_evpRevFac ) * denom2 * vs.iceMask

    # recover components of the stress tensor
    sig11 = 0.5 * ( sigma1 + sigma2 )
    sig22 = 0.5 * ( sigma1 - sigma2 )

    # calculate adaptive relaxation parameter on z-points
    if sett.useAdaptiveEVP:
        evpAlphaZ = 0.5 * (evpAlphaC + npx.roll(evpAlphaC, 1, 1))
        evpAlphaZ = 0.5 * (evpAlphaZ + npx.roll(evpAlphaZ, 1, 0))
        denom2 = 1.0 / evpAlphaZ

    # step sigma12
    sigma12 = ( sigma12 * ( evpAlphaZ - evpRevFac ) + shear * recip_evpRevFac ) * denom2

    # calculate divergence of stress tensor
    stressDivX, stressDivY = stressdiv(state, sig11, sig22, sigma12)

    # calculate drag coefficients
    cDrag = ocean_drag_coeffs(state, uIce, vIce)
    cBotC = basal_drag_coeffs(state, uIce, vIce)
    
    # over open ocean..., see comments in MITgcm: pkg/seaice/seaice_evp.F
    locMaskU = vs.SeaIceMassU
    locMaskV = vs.SeaIceMassV
    locMaskU = npx.where(locMaskU != 0, 1, locMaskU)
    locMaskV = npx.where(locMaskV != 0, 1, locMaskV)

    # calculate total ocean and wind forcing
    ForcingX = vs.WindForcingX + \
        ( 0.5 * ( cDrag + npx.roll(cDrag, 1, 1) ) * sett.cosWat * vs.uOcean
         - npx.sign(vs.fCori) * sett.sinWat * 0.5 * (
             cDrag * 0.5 * (
                 vs.uOcean - uIce
                 + npx.roll(vs.uOcean - uIce, -1, 0) )
             + npx.roll(cDrag, 1, 1) * 0.5 * (
                 npx.roll(vs.uOcean - uIce, 1, 1)
                 + npx.roll(npx.roll(vs.uOcean - uIce, 1, 1), -1, 0) )
             ) * locMaskU ) * vs.AreaW

    ForcingY = vs.WindForcingY + \
        ( 0.5 * ( cDrag + npx.roll(cDrag, 1, 0) ) * sett.cosWat * vs.vOcean
         + npx.sign(vs.fCori) * sett.sinWat * 0.5 * (
             cDrag * 0.5 * (
                 vs.vOcean - vIce
                 + npx.roll(vs.vOcean - vIce, -1, 1) )
             + npx.roll(cDrag, 1, 0) * 0.5 * (
                 npx.roll(vs.vOcean - vIce, 1, 0)
                 + npx.roll(npx.roll(vs.vOcean - vIce, 1, 0), -1, 1) )
             ) * locMaskV ) * vs.AreaS

    # add coriolis term
    fvAtC = vs.SeaIceMassC * vs.fCori * 0.5 * (vIce + npx.roll(vIce, -1, 0))
    fuAtC = vs.SeaIceMassC * vs.fCori * 0.5 * (uIce + npx.roll(uIce, -1, 1))
    ForcingX = ForcingX + 0.5 * (fvAtC + npx.roll(fvAtC, 1, 1))
    ForcingY = ForcingY - 0.5 * (fuAtC + npx.roll(fuAtC, 1, 0))

    # interpolate relaxation parameters to velocity points
    if sett.useAdaptiveEVP:
        evpBetaU = 0.5 * (evpAlphaC + npx.roll(evpAlphaC, 1, 1))
        evpBetaV = 0.5 * (evpAlphaC + npx.roll(evpAlphaC, 1, 0))

    betaFacU = evpBetaU * sett.recip_deltatDyn
    betaFacV = evpBetaV * sett.recip_deltatDyn
    betaFacP1U = betaFacU + sett.recip_deltatDyn
    betaFacP1V = betaFacV + sett.recip_deltatDyn
    
    denomU = vs.SeaIceMassU * betaFacP1U + 0.5 * (cDrag + npx.roll(cDrag, 1, 1)) * sett.cosWat * vs.AreaW
    denomV = vs.SeaIceMassV * betaFacP1V + 0.5 * (cDrag + npx.roll(cDrag, 1, 0)) * sett.cosWat * vs.AreaS
    
    denomU = denomU + vs.AreaW * 0.5 * (cBotC + npx.roll(cBotC, 1, 1))
    denomV = denomV + vs.AreaS * 0.5 * (cBotC + npx.roll(cBotC, 1, 0))
    
    denomU = npx.where(denomU==0,1,denomU)
    denomV = npx.where(denomV==0,1,denomV)
    
    # add lateral drag
    if sett.noSlip == False:
        SideDragU, SideDragV = side_drag(state, uIce, vIce)
        
        # the side drag coefficients are not multiplied by the area because they are calculated from
        # SeaIceMass which is calculated from hIceMean which includes the area
        denomU = denomU + SideDragU
        denomV = denomV + SideDragV

    uIce = vs.iceMaskU * (
        betaFacU * vs.SeaIceMassU * uIce
        + vs.SeaIceMassU * sett.recip_deltatDyn * uIceNm1
        + ForcingX + stressDivX
    ) / denomU
    vIce = vs.iceMaskV * (
        betaFacV * vs.SeaIceMassV * vIce
        + vs.SeaIceMassV * sett.recip_deltatDyn * vIceNm1
        + ForcingY + stressDivY
    ) / denomV

    # fill overlaps
    uIce, vIce = fill_overlap_uv(state, uIce, vIce)

    # residual computation
    if sett.computeEvpResidual:
        sig11Pm1 = (sigma11 - sig11Pm1) * evpAlphaC * vs.iceMask
        sig22Pm1 = (sigma22 - sig22Pm1) * evpAlphaC * vs.iceMask
        sig12Pm1 = (sigma12 - sig12Pm1) * evpAlphaZ  # * maskZ

        uIcePm1 = vs.iceMaskU * (uIce - uIcePm1) * evpBetaU
        vIcePm1 = vs.iceMaskV * (vIce - vIcePm1) * evpBetaV

        # if not explicitDrag:
        #     ForcingX = ForcingX - uIce * dragU
        #     ForcingY = ForcingY - vIce * dragV

        # uIcePm1 = ( SeaIceMassU * (uIce - uIceNm1)*recip_deltatDyn
        #             - (ForcingX + stressDivX)
        #            ) * iceMaskU
        # vIcePm1 = ( SeaIceMassV * (vIce - vIceNm1)*recip_deltatDyn
        #             - (ForcingY + stressDivY)
        #            ) * iceMaskV

        resSig = update(
            resSig,
            at[iEVP],
            (sig11Pm1**2 + sig22Pm1**2 + sig12Pm1**2)[
                sett.olx : -sett.olx, sett.oly : -sett.oly
            ].sum(),
        )
        resSig = update(resSig, at[iEVP], global_sum(resSig[iEVP]))
        resU = update(
            resU,
            at[iEVP],
            (uIcePm1**2 + vIcePm1**2)[
                sett.olx : -sett.olx, sett.oly : -sett.oly
            ].sum(),
        )
        resU = update(resU, at[iEVP], global_sum(resU[iEVP]))

        resEVP = resU[iEVP]
        resEVP0 = resU[0]
        resEVP = resEVP / resEVP0

        if printEvpResidual:
            print("evp resU, resSigma: %i %e %e" % (iEVP, resU[iEVP], resSig[iEVP]))

    return (
        state,
        uIce,
        vIce,
        uIceNm1,
        vIceNm1,
        sigma1,
        sigma2,
        sigma12,
        denom1,
        denom2,
        EVPcFac,
        evpAlphaC,
        evpAlphaZ,
        evpBetaU,
        evpBetaV,
        resSig,
        resU,
    )


@veros_kernel
def evp_solver(state):
    """solve the momentum equation and calculate u^n, sigma^n from u^(n-1), sigma^(n-1)
    using subcycling iterations of evp_solver_body
    """

    vs = state.variables
    settings = state.settings

    # calculate parameter used for adaptive relaxation parameters
    if settings.useAdaptiveEVP:
        aEVPcStar = 4
        EVPcFac = state.settings.deltatDyn * aEVPcStar * (npx.pi * settings.aEvpCoeff) ** 2
    else:
        EVPcFac = 0

    denom1 = npx.ones_like(vs.iceMask) / settings.evpAlpha
    denom2 = denom1

    # copy previous time step (n-1) of ice velocities and stress tensor
    uIceNm1 = vs.uIce
    vIceNm1 = vs.vIce
    uIce = vs.uIce
    vIce = vs.vIce
    sigma1 = vs.sigma1
    sigma2 = vs.sigma2
    sigma12 = vs.sigma12

    # initialize adaptive EVP specific fields
    evpAlphaC = npx.ones_like(vs.iceMask) * settings.evpAlpha
    evpAlphaZ = npx.ones_like(vs.iceMask) * settings.evpAlpha
    evpBetaU = npx.ones_like(vs.iceMask) * settings.evpBeta
    evpBetaV = npx.ones_like(vs.iceMask) * settings.evpBeta

    resSig = npx.zeros(settings.nEVPsteps)
    resU = npx.zeros(settings.nEVPsteps)

    # set argument for the loop (the for_loop of jax can only take one argument)
    arg_body = (
        state,
        uIce,
        vIce,
        uIceNm1,
        vIceNm1,
        sigma1,
        sigma2,
        sigma12,
        denom1,
        denom2,
        EVPcFac,
        evpAlphaC,
        evpAlphaZ,
        evpBetaU,
        evpBetaV,
        resSig,
        resU,
    )

    # calculate u^n, sigma^n and residuals
    arg_body = for_loop(0, settings.nEVPsteps, evp_solver_body, arg_body)

    # return uIce, vIce, sigma1, sigma2, sigma12
    return arg_body[1], arg_body[2], arg_body[5], arg_body[6], arg_body[7]



    
    
