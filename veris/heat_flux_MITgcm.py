import veris.heat_flux_constants as ct
from veros import veros_kernel
#from veros.variables import allocate
from veros.core.operators import numpy as npx # , update, at

# heat flux bulk formula of the MITgcm, used in the setup file of Veros

@veros_kernel
def bulkf_formula_lanl(uw, vw, ta, qa, tsf, ocn_mask):
    """Calculate bulk formula fluxes over open ocean

        wind stress = (ust,vst) = rhoA * Cd * Ws * (del.u,del.v)
        Sensib Heat flux = fsha = rhoA * Ch * Ws * del.T * CpAir
        Latent Heat flux = flha = rhoA * Ce * Ws * del.Q * Lvap
                        = -Evap * Lvap
        with Ws = wind speed = sqrt(del.u^2 +del.v^2) ;
            del.T = Tair - Tsurf ; del.Q = Qair - Qsurf
            Cd,Ch,Ce = transfer coefficient for momentum, sensible
                    & latent heat flux [no units]

    Arguments:
        uw (:obj:`ndarray`): zonal wind speed (at grid center) [m/s]
        vw (:obj:`ndarray`): meridional wind speed (at grid center) [m/s]
        ta (:obj:`ndarray`): air temperature   [K]     at height ht
        qa (:obj:`ndarray`): specific humidity [kg/kg] at heigth ht
        tsf(:obj:`ndarray`): sea surface temperature [K]
        ocn_mask (:obj:`ndarray`): 0=land, 1=ocean

    Returns:
        flwupa (:obj:`ndarray`): upward long wave radiation (>0 upward) [W/m2]
        flha   (:obj:`ndarray`): latent heat flux         (>0 downward) [W/m2]
        fsha   (:obj:`ndarray`): sensible heat flux       (>0 downward) [W/m2]
        df0dT  (:obj:`ndarray`): derivative of heat flux with respect to Tsf [W/m2/K]
        ust    (:obj:`ndarray`): zonal wind stress (at grid center)     [N/m2]
        vst    (:obj:`ndarray`): meridional wind stress (at grid center)[N/m2]
        evp    (:obj:`ndarray`): evaporation rate (over open water) [kg/m2/s]
        ssq    (:obj:`ndarray`): surface specific humidity          [kg/kg]
        dEvdT  (:obj:`ndarray`): derivative of evap. with respect to tsf [kg/m2/s/K]
    """

    # Compute turbulent surface fluxes
    ht =  2.
    zref = 10.
    zice = 0.0005
    aln = npx.log(ht / zref)
    czol = zref * ct.KARMAN * ct.G

    lath = npx.ones_like(ocn_mask) * ct.LATVAP

    # wind speed
    us = npx.sqrt(uw[...] * uw[...] + vw[...] * vw[...])
    usm = npx.maximum(us[...], 1.0)

    t0 = ta[...] * (1.0 + ct.ZVIR * qa[...])
    ssq = 3.797915 * npx.exp(lath[...] * (7.93252e-6 - 2.166847e-3 / tsf[...])) / 1013.

    deltap = ta[...] - tsf[...] + ct.GAMMA_BLK * ht
    delq = qa[...] - ssq[...]

    # initialize estimate exchange coefficients
    rdn = ct.KARMAN / npx.log(zref / zice)
    rhn = rdn
    ren = rdn
    # calculate turbulent scales
    ustar = rdn * usm[...]
    tstar = rhn * deltap[...]
    qstar = ren * delq[...]

    # iteration with psi-functions to find transfer coefficients
    for _ in range(5):
        huol = czol / ustar[...]**2 * (tstar[...] / t0 + qstar[...]/(1. / ct.ZVIR + qa[...]))
        huol = npx.minimum(npx.abs(huol[...]), 10.0) * npx.sign(huol[...])
        stable = 0.5 + 0.5 * npx.sign(huol[...])
        xsq = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * huol[...])), 1.0)
        x = npx.sqrt(xsq[...])
        psimh = -5. * huol[...] * stable[...] + (1. - stable[...])\
              * (2. * npx.log(0.5 * (1. + x[...]))
                + 2. * npx.log(0.5 * (1. + xsq[...]))
                - 2. * npx.arctan(x[...]) + npx.pi * 0.5)
        psixh = -5. * huol[...] * stable[...] + (1. - stable[...])\
              *  (2. * npx.log(0.5 * (1. + xsq[...])))

        # update the transfer coefficients
        rd = rdn / (1. + rdn * (aln[...] - psimh[...]) / ct.KARMAN)
        rh = rhn / (1. + rhn * (aln[...] - psixh[...]) / ct.KARMAN)
        re = rh

        # update ustar, tstar, qstar using updated, shifted coefficients.
        ustar = rd[...] * usm[...]
        qstar = re[...] * delq[...]
        tstar = rh[...] * deltap[...]

    #tau = ct.RHOA * ustar[...]**2
    #tau = tau * us[...] / usm[...]
    csha = ct.RHOA * ct.CPDAIR * us[...] * rh[...] * rd[...]
    clha = ct.RHOA * lath[...] * us[...] * re[...] * rd[...]

    fsha = csha[...] * deltap[...]
    flha = clha[...] * delq[...]
    evp = -flha[...] / lath[...]

    flwupa = ct.OCEAN_EMISSIVITY * ct.STEBOL * tsf[...]**4
    dflwupdt = 4. * ct.OCEAN_EMISSIVITY * ct.STEBOL * tsf[...]**3

    devdt = clha[...] * ssq[...] * 2.166847e-3 / (tsf[...] * tsf[...])
    dflhdt = -lath[...] * devdt[...]
    dfshdt = -csha[...]

    # total derivative with respect to surface temperature
    df0dt = -dflwupdt[...] + dfshdt[...] + dflhdt[...]

    #  wind stress at center points
    bulkf_cdn = 2.7e-3 / usm[...] + 0.142e-3 + 0.0764e-3 * usm[...]
    ust = ct.RHOA * bulkf_cdn * us[...] * uw[...]
    vst = ct.RHOA * bulkf_cdn * us[...] * vw[...]

    return (flwupa, flha, fsha, df0dt, ust, vst, evp, ssq, devdt)