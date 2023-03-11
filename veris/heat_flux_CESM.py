import veris.heat_flux_constants as ct
from veros import veros_kernel
# from veros.variables import allocate
from veros.core.operators import numpy as npx, update, at

# heat flux bulk formula of the CESM, used in the setup file of Veros

_cc = npx.array([0.88, 0.84, 0.80,
                0.76, 0.72, 0.68,
                0.63, 0.59, 0.52,
                0.50, 0.50, 0.50,
                0.52, 0.59, 0.63,
                0.68, 0.72, 0.76,
                0.80, 0.84, 0.88])

_clat = npx.array([-90.0, -80.0, -70.0,
                    -60.0, -50.0, -40.0,
                    -30.0, -20.0, -10.0,
                    -5.0,   0.0,   5.0,
                    10.0,  20.0,  30.0,
                    40.0,  50.0,  60.0,
                    70.0,  80.0,  90.0])

@veros_kernel
def qsat(tk):
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    return 640380.0 / npx.exp(5107.4 / tk)

@veros_kernel
def qsat_august_eqn(ps, tk):
    """Saturated specific humidity (kg/kg)

    Arguments:
        ps (:obj:`ndarray`): atm sfc pressure (Pa)
        tk (:obj:`ndarray`): atm temperature (K)

    Returns:
        :obj:`ndarray`

    Reference:
        Barnier B., L. Siefridt, P. Marchesiello, (1995):
        Thermal forcing for a global ocean circulation model
        using a three-year climatology of ECMWF analyses,
        Journal of Marine Systems, 6, p. 363-380.
    """
    return 0.622 / ps * 10**(9.4051 - 2353. / tk) * 133.322

@veros_kernel
def dqnetdt(mask, ps, rbot, sst, ubot, vbot, us, vs):
    """Calculates correction term of net ocean heat flux (W/m^2)

    Arguments:
        mask (:obj:`ndarray`): ocean mask (0-1)
        ps (:obj:`ndarray`): surface pressure (Pa)
        rbot (:obj:`ndarray`): atm density at full model level (kg/m^3)
        sst (:obj:`ndarray`): surface temperature (K)
        vmag (:obj:`ndarray`): atm wind speed at full model level (m/s)

    Returns:
        tuple(:obj:`ndarray`, :obj:`ndarray`, :obj:`ndarray`)

    Reference:
        Barnier B., L. Siefridt, P. Marchesiello, (1995):
        Thermal forcing for a global ocean circulation model
        using a three-year climatology of ECMWF analyses,
        Journal of Marine Systems, 6, p. 363-380.
    """

    vmag = npx.maximum(ct.UMIN_O, npx.sqrt((ubot[...] - us[...])**2
                                        + (vbot[...] - vs[...])**2))

    dqir_dt = -ct.STEBOL * 4. * sst[...]**3 * mask  # long-wave radiation correction (IR)
    dqh_dt = -rbot[...] * ct.CPDAIR * ct.CH * vmag[...] * mask  # sensible heat flux correction
    dqe_dt = -rbot[...] * ct.CE * ct.LATVAP * vmag[...] * 2353.\
        * npx.log(10.) * qsat_august_eqn(ps, sst) / (sst[...]**2) * mask  # latent heat flux correction

    return (dqir_dt, dqh_dt, dqe_dt)

@veros_kernel
def net_lw_ocn(state, mask, lat, qbot, sst, tbot, tcc):
    """Compute net LW (upward - downward) radiation at the ocean surface (W/m^2)

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask        0 <=> out of domain
        lat (:obj:`ndarray`): latitude coordinates    (deg)
        qbot (:obj:`ndarray`): atm specific humidity  (kg/kg)
        sst (:obj:`ndarray`): sea surface temperature (K)
        tbot (:obj:`ndarray`): atm T                  (K)
        tcc (:obj:`ndarray`): total cloud cover       (0-1)

    Returns:
        :obj:`ndarray`

    Reference:
        Clark, N.E., L.Eber, R.M.Laurs, J.A.Renner, and J.F.T.Saur, (1974):
        Heat exchange between ocean and atmosphere in the eastern North Pacific for 1961-71,
        NOAA Technical report No. NMFS SSRF-682.
    """

    ccint = npx.zeros(lat.shape)
    idx_num = npx.arange(lat.size)
    #ccint = allocate(state.dimensions, ("yt",))

    for i in range(20):
        #idx = npx.squeeze(npx.argwhere((lat[:] > _clat[i]) & (lat[:] <= _clat[i+1])))
        idx = npx.where((lat[:] > _clat[i]) & (lat[:] <= _clat[i+1]), idx_num, 0) # to make it work with JAX
        ccint = update(ccint, at[idx],
                        _cc[i] + (_cc[i+1] - _cc[i])\
                       * (lat[idx] - _clat[i]) / (_clat[i+1] - _clat[i])
        )

    frac_cloud_cover = 1. - ccint[npx.newaxis, :] * tcc[...]**2
    rtea = npx.sqrt(1000. * qbot[...] / (0.622 + 0.378 * qbot[...]) + ct.EPS2)

    return -ct.EMISSIVITY * ct.STEBOL * tbot[...]**3\
        * (tbot[...] * (0.39 - 0.05 * rtea[...]) * frac_cloud_cover
           + 4. * (sst[...] - tbot[...])) * mask[...]

@veros_kernel
def cdn(umps):
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    return 0.0027 / umps + 0.000142 + 0.0000764 * umps

@veros_kernel
def psimhu(xd):
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return npx.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)\
        - 2.0 * npx.arctan(xd) + 1.571

@veros_kernel
def psixhu(xd):
    """Unstable part of psimx

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return 2.0 * npx.log((1.0 + xd * xd) / 2.0)

@veros_kernel
def flux_atmOcn(mask, rbot, zbot, ubot, vbot, qbot, tbot, thbot, us, vs, ts):
    """atm/ocn fluxes calculation

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        rbot (:obj:`ndarray`): atm density           (kg/m^3)
        zbot (:obj:`ndarray`): atm level height      (m)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        tbot (:obj:`ndarray`): atm T                 (K)
        thbot(:obj:`ndarray`): atm potential T       (K)
        us   (:obj:`ndarray`): ocn u-velocity        (m/s)
        vs   (:obj:`ndarray`): ocn v-velocity        (m/s)
        ts   (:obj:`ndarray`): ocn temperature       (K)

    Returns:
        sen  (:obj:`ndarray`): heat flux: sensible    (W/m^2)
        lat  (:obj:`ndarray`): heat flux: latent      (W/m^2)
        lwup (:obj:`ndarray`): heat flux: lw upward   (W/m^2)
        evap (:obj:`ndarray`): water flux: evap  ((kg/s)/m^2)
        taux (:obj:`ndarray`): surface stress, zonal      (N)
        tauy (:obj:`ndarray`): surface stress, maridional (N)

        tref (:obj:`ndarray`): diag:  2m ref height T     (K)
        qref (:obj:`ndarray`): diag:  2m ref humidity (kg/kg)
        duu10n(:obj:`ndarray`): diag: 10m wind speed squared (m/s)^2

        ustar_sv(:obj:`ndarray`): diag: ustar
        re_sv   (:obj:`ndarray`): diag: sqrt of exchange coefficient (water)
        ssq_sv  (:obj:`ndarray`): diag: sea surface humidity  (kg/kg)

    Reference:
        - Large, W. G., & Pond, S. (1981). Open Ocean Momentum Flux Measurements in Moderate to Strong Winds,
        Journal of Physical Oceanography, 11(3), pp. 324-336
        - Large, W. G., & Pond, S. (1982). Sensible and Latent Heat Flux Measurements over the Ocean,
        Journal of Physical Oceanography, 12(5), 464-482.
        - https://svn-ccsm-release.cgd.ucar.edu/model_versions/cesm1_0_5/models/csm_share/shr/shr_flux_mod.F90
    """

    al2 = npx.log(ct.ZREF / ct.ZTREF)

    vmag = npx.maximum(ct.UMIN_O, npx.sqrt((ubot[...] - us[...])**2
                                        + (vbot[...] - vs[...])**2))

    # sea surface humidity (kg/kg)
    ssq = 0.98 * qsat(ts[...]) / rbot[...]

    # potential temperature diff. (K)
    delt = thbot[...] - ts[...]

    # specific humidity diff. (kg/kg)
    delq = qbot[...] - ssq[...]

    alz = npx.log(zbot[...] / ct.ZREF)
    cp = ct.CPDAIR * (1.0 + ct.CPVIR * ssq[...])

    # first estimate of Z/L and ustar, tstar and qstar

    # neutral coefficients, z/L = 0.0
    stable = 0.5 + 0.5 * npx.sign(delt[...])
    rdn = npx.sqrt(cdn(vmag[...]))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = 0.0346

    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thbot[...] + qstar[...]
        / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol = npx.minimum(npx.abs(hol[...]), 10.0) * npx.sign(hol[...])
    stable = 0.5 + 0.5 * npx.sign(hol[...])
    xsq = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = npx.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = npx.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / ct.KARMAN * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # iterate to converge on Z/L, ustar, tstar and qstar

    # compute stability & evaluate all stability functions
    hol = ct.KARMAN * ct.G * zbot[...] *\
        (tstar[...] / thbot[...] + qstar[...]
         / (1.0 / ct.ZVIR + qbot[...])) / ustar[...]**2
    hol = npx.minimum(npx.abs(hol[...]), 10.0) * npx.sign(hol[...])
    stable = 0.5 + 0.5 * npx.sign(hol[...])
    xsq = npx.maximum(npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = npx.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = npx.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / ct.KARMAN * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / ct.KARMAN * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / ct.KARMAN * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # compute the fluxes

    tau = rbot[...] * ustar[...] * ustar[...]

    # momentum flux
    taux = tau[...] * (ubot[...] - us[...]) / vmag[...] * mask[...]
    tauy = tau[...] * (vbot[...] - vs[...]) / vmag[...] * mask[...]

    # heat flux
    sen = cp[...] * tau[...] * tstar[...] / ustar[...] * mask[...]
    lat = ct.LATVAP * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -ct.STEBOL * ts[...]**4 * mask[...]

    # water flux
    evap = lat[...] / ct.LATVAP * mask[...]

    # compute diagnositcs: 2m ref T & Q, 10m wind speed squared

    hol = hol[...] * ct.ZTREF / zbot[...]
    xsq = npx.maximum(1.0, npx.sqrt(npx.abs(1.0 - 16.0 * hol[...])))
    xqq = npx.sqrt(xsq)
    psix2 = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])
    fac = (rh[...] / ct.KARMAN) * (alz[...] + al2 - psixh[...] + psix2[...])
    tref = thbot[...] - delt[...] * fac[...]

    # pot. temp to temp correction
    tref = (tref[...] - 0.01 * ct.ZTREF) * mask[...]
    fac = (re[...] / ct.KARMAN) * (alz[...] + al2 - psixh[...] + psix2[...]) * mask[...]
    qref = (qbot[...] - delq[...] * fac[...]) * mask[...]

    # 10m wind speed squared
    duu10n = u10n[...] * u10n[...] * mask[...]

    return (sen, lat, lwup, evap, taux, tauy, tref, qref, duu10n, ustar, tstar, qstar)

@veros_kernel
def flux_atmOcn_simple(mask, ps, qbot, rbot, ubot, vbot, tbot, us, vs, ts):
    """Calculates bulk net heat flux

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        ps   (:obj:`ndarray`): surface pressure (Pa)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        rbot (:obj:`ndarray`): atm density at full model level (kg/m^3)
        tbot (:obj:`ndarray`): temperature at full model level (K)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        us   (:obj:`ndarray`): ocn u-velocity        (m/s)
        vs   (:obj:`ndarray`): ocn v-velocity        (m/s)
        ts   (:obj:`ndarray`): surface temperature   (K)

    Returns:
        tuple(:obj:`ndarray`, :obj:`ndarray`, :obj:`ndarray`) 

    Reference:
        Barnier B., L. Siefridt, P. Marchesiello, (1995):
        Thermal forcing for a global ocean circulation model
        using a three-year climatology of ECMWF analyses,
        Journal of Marine Systems, 6, p. 363-380.
    """

    vmag = npx.maximum(ct.UMIN_O, npx.sqrt((ubot[...] - us[...])**2
                                        + (vbot[...] - vs[...])**2))

    # long-wave radiation (IR)
    qir = -ct.STEBOL * ts[...]**4 * mask[...]

    # sensible heat flux
    qh = rbot[...] * ct.CPDAIR * ct.CH * vmag[...] * (tbot[...] - ts[...]) * mask[...]

    # latent heat flux
    qe = -rbot[...] * ct.CE * ct.LATVAP * vmag[...] * (qsat_august_eqn(ps, ts)
                                                       - qbot[...]) * mask[...]

    return (qir, qh, qe)