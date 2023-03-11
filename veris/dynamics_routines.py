from veros.core.operators import numpy as npx
from veros import veros_kernel

@veros_kernel
def SeaIceStrength(state):

    '''calculate ice strength (= maximum compressive stress)
    from ice thickness and ice cover fraction
    '''

    vs = state.variables
    sett = state.settings

    SeaIceStrength = sett.pStar * vs.hIceMean * npx.exp(-sett.cStar * (1 - vs.Area)) * vs.iceMask

    return SeaIceStrength

@veros_kernel
def ocean_drag_coeffs(state, uIce, vIce):

    '''calculate linear ice-water drag coefficient from ice and ocean velocities
    (this coefficient creates a linear relationship between
    ice-ocean stress difference and ice-ocean velocity difference)
    '''

    vs = state.variables
    sett = state.settings

    # get ice-water drag coefficient times density
    dragCoeff = npx.where(vs.fCori < 0, sett.waterIceDrag_south, sett.waterIceDrag) * sett.rhoSea

    # calculate component-wise velocity differences at velocity points
    du = (uIce - vs.uOcean) * vs.maskInU
    dv = (vIce - vs.vOcean) * vs.maskInV

    # calculate total velocity difference at c-point
    tmpVar = 0.25 * ( (du + npx.roll(du,-1,0))**2
                    +  (dv + npx.roll(dv,-1,1))**2 )

    # calculate linear drag coefficient and apply mask
    cDrag = npx.where(dragCoeff**2 * tmpVar > sett.cDragMin**2,
                        dragCoeff * npx.sqrt(tmpVar), sett.cDragMin)
    cDrag = cDrag * vs.iceMask

    return cDrag

@veros_kernel
def basal_drag_coeffs(state, uIce, vIce):

    '''calculate basal drag coefficient to account for the formation of 
    landfast ice in shallow waters due to the formation of ice keels
    (Lemieux et al., 2015)
    '''

    vs = state.variables
    sett = state.settings

    # absolute value of the ice velocity at c-points
    tmpFld = 0.25 * ( (uIce * vs.maskInU)**2
                    + npx.roll(uIce * vs.maskInU,-1,0)**2
                    + (vIce * vs.maskInV)**2
                    + npx.roll(vIce * vs.maskInV,-1,1)**2 )

    # include velocity parameter U0 to avoid singularities
    tmpFld = sett.basalDragK2 / npx.sqrt(tmpFld + sett.basalDragU0**2)

    # critical ice height that allows for the formation of landfast ice
    hCrit = npx.abs(vs.R_low) * vs.Area / sett.basalDragK1

    # soft maximum for better differentiability:
    # max(a,b;k) = ln(exp(k*a)+exp(k*b))/k
    # In our case, b=0, so exp(k*b) = 1.
    # max(a,0;k) = ln(exp(k*a)+1)/k
    # If k*a gets too large, EXP will overflow, but for the anticipated
    # values of hActual < 100m, and k=10, this should be very unlikely
    fac = 10. 
    recip_fac = 1. / fac
    cBot = npx.where(vs.Area > 0.01,
            tmpFld
                * npx.log(npx.exp(fac * (vs.hIceMean-hCrit)) + 1.)
                * recip_fac * npx.exp(-sett.cBasalStar * (1. - vs.Area)),
                0.)

    return cBot

@veros_kernel
def side_drag(state, uIce, vIce):

    '''calculate the lateral drag coefficient to simulate landfast ice
    (Liu et al. 2022, A new parameterization of coastal drag to simulate landfast
    ice in deep marginal seas in the Arctic)
    '''

    vs = state.variables
    sett = state.settings

    # calculate total ice speed at c-points
    iceSpeed = 0.5 * npx.sqrt(  (uIce + npx.roll(uIce,-1,0))**2
                            +   (vIce + npx.roll(vIce,-1,1))**2  )

    # interpolate total ice speed to u- and v-points
    iceSpeedU = npx.where(vs.AreaW > 0, 0.5 * (iceSpeed + npx.roll(iceSpeed,1,0)), 0)
    iceSpeedV = npx.where(vs.AreaS > 0, 0.5 * (iceSpeed + npx.roll(iceSpeed,1,1)), 0)

    # these masks give the number of neighbouring land cells in u- and v-direction
    # (can only be non-zero for ocean cells)
    maskU = ( 2 - npx.roll(vs.iceMaskU,1,1)
                - npx.roll(vs.iceMaskU,-1,1)
                ) * vs.iceMaskU
    maskV = ( 2 - npx.roll(vs.iceMaskV,1,0)
                - npx.roll(vs.iceMaskV,-1,0)
                ) * vs.iceMaskV

    # use the coastline to determine the form factor
    if sett.use_coastline:
        maskU = vs.Fu
        maskV = vs.Fv

    # calculate side drag coefficients
    SideDragU = vs.SeaIceMassU * sett.sideDragCoeff * maskU / (iceSpeedU + sett.sideDragU0)
    SideDragV = vs.SeaIceMassV * sett.sideDragCoeff * maskV / (iceSpeedV + sett.sideDragU0)

    return SideDragU, SideDragV

@veros_kernel
def strainrates(state, uIce, vIce):

    '''calculate strain rate tensor components from ice velocities'''

    vs = state.variables
    sett = state.settings

    # some abbreviations at c-points
    dudx = ( npx.roll(uIce,-1,axis=0) - uIce ) * vs.recip_dxU
    uave = ( npx.roll(uIce,-1,axis=0) + uIce ) * 0.5
    dvdy = ( npx.roll(vIce,-1,axis=1) - vIce ) * vs.recip_dyV
    vave = ( npx.roll(vIce,-1,axis=1) + vIce ) * 0.5

    # calculate strain rates at c-points
    e11 = ( dudx + vave * vs.k2AtC ) * vs.maskInC
    e22 = ( dvdy + uave * vs.k1AtC ) * vs.maskInC

    # some abbreviations at z-points
    dudy = ( uIce - npx.roll(uIce,1,axis=1) ) * vs.recip_dyU
    uave = ( uIce + npx.roll(uIce,1,axis=1) ) * 0.5
    dvdx = ( vIce - npx.roll(vIce,1,axis=0) ) * vs.recip_dxV
    vave = ( vIce + npx.roll(vIce,1,axis=0) ) * 0.5

    # calculate strain rate at z-points
    mskZ = vs.iceMask*npx.roll(vs.iceMask,1,axis=0)
    mskZ =    mskZ*npx.roll(   mskZ,1,axis=1)
    e12 = 0.5 * ( dudy + dvdx - vs.k1AtZ * vave - vs.k2AtZ * uave ) * mskZ
    if sett.noSlip:
        hFacU = vs.iceMaskU - npx.roll(vs.iceMaskU,1,axis=1)
        hFacV = vs.iceMaskV - npx.roll(vs.iceMaskV,1,axis=0)
        e12   = e12 + ( 2.0 * uave * vs.recip_dyU * hFacU
                      + 2.0 * vave * vs.recip_dxV * hFacV )

    if sett.noSlip and sett.secondOrderBC:
        hFacU = ( vs.iceMaskU - npx.roll(vs.iceMaskU,1,1) ) / 3.
        hFacV = ( vs.iceMaskV - npx.roll(vs.iceMaskV,1,0) ) / 3.
        hFacU = hFacU * (npx.roll(vs.iceMaskU, 2,1) * npx.roll(vs.iceMaskU,1,1)
                       + npx.roll(vs.iceMaskU,-1,1) * vs.iceMaskU )
        hFacV = hFacV * (npx.roll(vs.iceMaskV, 2,0) * npx.roll(vs.iceMaskV,1,0)
                       + npx.roll(vs.iceMaskV,-1,0) * vs.iceMaskV )
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
            vs.recip_dyU * ( 6. * uave
                          - npx.roll(uIce, 2,1) * npx.roll(vs.iceMaskU,1,1)
                          - npx.roll(uIce,-1,1) * vs.iceMaskU ) * hFacU
          + vs.recip_dxV * ( 6. * vave
                          - npx.roll(vIce, 2,0) * npx.roll(vs.iceMaskV,1,0)
                          - npx.roll(vIce,-1,0) * vs.iceMaskV ) * hFacV
        )

    return e11, e22, e12

@veros_kernel
def viscosities(state,e11,e22,e12):

    """calculate bulk viscosity zeta, shear viscosity eta, and ice pressure
    from strain rate tensor components and ice strength.
    if pressReplFac = 1, a replacement pressure is used to avoid
    stresses without velocities (Hibler and Ib, 1995).
    with tensileStrFac != 1, a resistance to tensile stresses can be included
    (König Beatty and Holland, 2010).
    """

    vs = state.variables
    sett = state.settings

    recip_PlasDefCoeffSq = 1. / sett.PlasDefCoeff**2

    # interpolate squares of e12 to c-points after weighting them with the
    # area centered around z-points
    e12Csq = vs.rAz * e12**2
    e12Csq =                     e12Csq + npx.roll(e12Csq,-1,1)
    e12Csq = 0.25 * vs.recip_rA * ( e12Csq + npx.roll(e12Csq,-1,0) )

    # calculate Delta from the normal strain rate (e11+e22)
    # and the shear strain rate sqrt( (e11-e22)**2 + 4 * e12**2) )
    deltaSq = (e11+e22)**2 + recip_PlasDefCoeffSq * (
        (e11-e22)**2 + 4. * e12Csq )
    deltaC = npx.sqrt(deltaSq)

    # use regularization to avoid singularies of zeta
    # TODO implement smooth regularization after comparing with the MITgcm
    # smooth regularization of delta for better differentiability
    # deltaCreg = deltaC + deltaMin
    # deltaCreg = npx.sqrt( deltaSq + deltaMin**2 )
    deltaCreg = npx.maximum(deltaC,sett.deltaMin)

    # calculate viscosities
    zeta = 0.5 * ( vs.SeaIceStrength * (1 + sett.tensileStrFac) ) / deltaCreg
    eta  = zeta * recip_PlasDefCoeffSq

    # calculate ice pressure
    press = 0.5 * ( vs.SeaIceStrength * (1 - sett.pressReplFac)
              + 2. * zeta * deltaC * sett.pressReplFac / (1 + sett.tensileStrFac)
             ) * (1 - sett.tensileStrFac)

    return zeta, eta, press

@veros_kernel
def stress(state,e11, e22, e12, zeta, eta, press):

    '''calculate stress tensor components'''

    from veris.averaging import c_point_to_z_point
    sig11 = zeta*(e11 + e22) + eta*(e11 - e22) - press
    sig22 = zeta*(e11 + e22) - eta*(e11 - e22) - press
    sig12 = 2. * e12 * c_point_to_z_point(state,eta)

    return sig11, sig22, sig12

@veros_kernel
def stressdiv(state,sig11, sig22, sig12):

    '''calculate divergence of stress tensor'''

    vs = state.variables

    stressDivX = (
          sig11*vs.dyV - npx.roll(sig11*vs.dyV, 1,axis=0)
        - sig12*vs.dxV + npx.roll(sig12*vs.dxV,-1,axis=1)
    ) * vs.recip_rAu
    stressDivY = (
          sig22*vs.dxU - npx.roll(sig22*vs.dxU, 1,axis=1)
        - sig12*vs.dyU + npx.roll(sig12*vs.dyU,-1,axis=0)
    ) * vs.recip_rAv

    return stressDivX, stressDivY