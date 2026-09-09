"""Standalone JAX bulk heat-flux kernels, retaining the original CESM equations.

Array inputs preserve the original shapes and units documented per function.
Settings and physical constants are supplied as separate frozen dataclasses.
The return casts describe JIT's array outputs for formulas whose eager NumPy
or Python inputs would otherwise infer NumPy arrays or scalars. They perform no
conversion and leave the original equations unchanged.
"""

from functools import partial
from typing import cast

import jax.numpy as npx
from jax import Array
from jax.typing import ArrayLike

from veris._bulk_types import CESMFluxes, HeatFluxes
from veris._typing import ArrayInput, MaskInput, jit
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants


@partial(jit, static_argnames=["phys"])
def qsat(phys: PhysicalConstants, tk: ArrayLike) -> Array:
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    return phys.cesmSaturationHumidityScale / npx.exp(
        phys.cesmSaturationHumidityTemperature / tk
    )


@partial(jit, static_argnames=["phys"])
def qsat_august_eqn(phys: PhysicalConstants, ps: ArrayLike, tk: ArrayLike) -> Array:
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
    return cast(
        Array,
        phys.waterVaporDryAirMassRatio
        / ps
        * 10
        ** (
            phys.augustVaporPressureLog10Offset
            - phys.augustVaporPressureTemperature / tk
        )
        * phys.mmHgToPa,
    )


@jit
def get_press_levs(sp: ArrayInput, hya: ArrayInput, hyb: ArrayInput) -> Array:
    """Compute pressure levels

    Arguments:
        sp (:obj:`ndarray`): Atmospheric surface pressure
        hya (:obj:`ndarray`): Hybrid sigma level A coefficient for vertical grid
        hyb (:obj:`ndarray`): Hybrid sigma level B coefficient for vertical grid

    Return:
        :obj:`ndarray`
    """

    return cast(
        Array,
        hya[npx.newaxis, npx.newaxis, :]
        + hyb[npx.newaxis, npx.newaxis, :] * sp[:, :, npx.newaxis],
    )


def compute_z_level(
    phys: PhysicalConstants, t: ArrayInput, q: ArrayInput, ph: ArrayInput
) -> Array:
    """Computes the altitudes at ECMWF Integrated Forecasting System
    (ECMWF-IFS) model half- and full-levels (for 137 levels model reanalysis: L137)

    Arguments:
        t (:obj:`ndarray`): Atmospheric temperture [K]
        q (:obj:`ndarray`): Atmospheric specific humidity [kg/kg]
        ph (:obj:`ndarray`): Pressure at half model levels

    Note:
        The top level of the atmosphere is excluded

    Reference:
        - https://www.ecmwf.int/sites/default/files/elibrary/2015/
        9210-part-iii-dynamics-and-numerical-procedures.pdf
        - https://confluence.ecmwf.int/display/CKB/
        ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height

    Returns:
        :obj:`ndarray`: Altitude of the atmospheric near surface layer (second IFS level)
    """

    # virtual temperature (K)
    tv = t[...] * (1.0 + phys.zvir * q[...])

    # compute geopotential for 2 lowermost (near-surface) model levels
    dlog_p = npx.log(ph[:, :, 1:] / ph[:, :, :-1])
    alpha = 1.0 - ((ph[:, :, :-1] / (ph[:, :, 1:] - ph[:, :, :-1])) * dlog_p)
    tv = tv * phys.rdair

    # zh is the geopotential of 'half-levels'
    # integrate zh to next half level
    increment = npx.flip(tv * dlog_p, axis=2)
    zh = npx.cumsum(increment, axis=2)

    # zf is the geopotential of this full level
    # integrate from previous (lower) half-level zh to the
    # full level
    increment_zh = npx.insert(zh, 0, 0, axis=2)
    zf = npx.flip(tv * alpha, axis=2) + increment_zh[:, :, :-1]

    alt = phys.radius * zf / phys.gravity / (phys.radius - zf / phys.gravity)

    return alt[:, :, -1]


@partial(jit, static_argnames=["sett", "phys"])
def dqnetdt(
    sett: Settings,
    phys: PhysicalConstants,
    mask: MaskInput,
    ps: ArrayInput,
    rbot: ArrayInput,
    sst: ArrayInput,
    ubot: ArrayInput,
    vbot: ArrayInput,
    us: ArrayInput,
    vs: ArrayInput,
) -> HeatFluxes:
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

    vmag = npx.maximum(
        sett.umin_o,
        npx.sqrt((ubot[...] - us[...]) ** 2 + (vbot[...] - vs[...]) ** 2),
    )

    # long-wave radiation correction (IR)
    dqir_dt = -phys.stefBoltz * 4.0 * sst[...] ** 3 * mask

    # sensible heat flux correction
    dqh_dt = -rbot[...] * phys.cpdair * phys.ch * vmag[...] * mask

    # latent heat flux correction
    dqe_dt = (
        -rbot[...]
        * phys.ce
        * phys.latvap
        * vmag[...]
        * phys.augustVaporPressureTemperature
        * npx.log(10.0)
        * qsat_august_eqn(phys, ps, sst)
        / (sst[...] ** 2)
        * mask
    )

    return cast(Array, dqir_dt), cast(Array, dqh_dt), cast(Array, dqe_dt)


@partial(jit, static_argnames=["sett", "phys"])
def net_lw_ocn(
    sett: Settings,
    phys: PhysicalConstants,
    mask: MaskInput,
    lat: ArrayInput,
    qbot: ArrayInput,
    sst: ArrayInput,
    tbot: ArrayInput,
    tcc: ArrayInput,
) -> Array:
    """Compute net downward LW radiation at the ocean surface (W/m^2)

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

    # Interpolate each latitude independently, including both polar endpoints.
    ccint = npx.interp(
        lat,
        npx.asarray(phys.longwaveCloudLatitudes),
        npx.asarray(phys.longwaveCloudCoefficients),
    )

    frac_cloud_cover = 1.0 - ccint[npx.newaxis, :] * tcc[...] ** 2
    rtea = npx.sqrt(
        phys.longwaveHumidityPressureScale
        * qbot[...]
        / (
            phys.waterVaporDryAirMassRatio
            + (1.0 - phys.waterVaporDryAirMassRatio) * qbot[...]
        )
        + sett.eps2
    )

    return cast(
        Array,
        -phys.emissivity
        * phys.stefBoltz
        * tbot[...] ** 3
        * (
            tbot[...]
            * (
                phys.longwaveClearSkyOffset
                - phys.longwaveHumidityCoefficient * rtea[...]
            )
            * frac_cloud_cover
            + 4.0 * (sst[...] - tbot[...])
        )
        * mask[...],
    )


@partial(jit, static_argnames=["phys"])
def cdn(phys: PhysicalConstants, umps: ArrayLike) -> Array:
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    return cast(
        Array,
        phys.neutralDragInverseWind / umps
        + phys.neutralDragConstant
        + phys.neutralDragLinearWind * umps,
    )


@partial(jit, static_argnames=["phys"])
def psimhu(phys: PhysicalConstants, xd: ArrayLike) -> Array:
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return (
        npx.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
        - 2.0 * npx.arctan(xd)
        + phys.cesmUnstableMomentumOffset
    )


@jit
def psixhu(xd: ArrayLike) -> Array:
    """Unstable part of psimx

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return 2.0 * npx.log((1.0 + xd * xd) / 2.0)


@partial(jit, static_argnames=["sett", "phys"])
def flux_atmOcn(
    sett: Settings,
    phys: PhysicalConstants,
    mask: MaskInput,
    rbot: ArrayInput,
    zbot: ArrayInput,
    ubot: ArrayInput,
    vbot: ArrayInput,
    qbot: ArrayInput,
    tbot: ArrayInput,
    thbot: ArrayInput,
    us: ArrayInput,
    vs: ArrayInput,
    ts: ArrayInput,
) -> CESMFluxes:
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

    al2 = npx.log(sett.zref / sett.ztref)

    vmag = npx.maximum(
        sett.umin_o,
        npx.sqrt((ubot[...] - us[...]) ** 2 + (vbot[...] - vs[...]) ** 2),
    )

    # sea surface humidity (kg/kg)
    ssq = phys.seawaterHumidityFactor * qsat(phys, ts[...]) / rbot[...]

    # potential temperature diff. (K)
    delt = thbot[...] - ts[...]

    # specific humidity diff. (kg/kg)
    delq = qbot[...] - ssq[...]

    alz = npx.log(zbot[...] / sett.zref)
    cp = phys.cpdair * (1.0 + phys.cpvir * ssq[...])

    # first estimate of Z/L and ustar, tstar and qstar

    # neutral coefficients, z/L = 0.0
    stable = 0.5 + 0.5 * npx.sign(delt[...])
    rdn = npx.sqrt(cdn(phys, vmag[...]))
    rhn = (
        1.0 - stable
    ) * phys.cesmNeutralHeatUnstable + stable * phys.cesmNeutralHeatStable
    ren = phys.cesmNeutralMoisture

    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # compute stability & evaluate all stability functions
    hol = (
        phys.karman
        * phys.gravity
        * zbot[...]
        * (tstar[...] / thbot[...] + qstar[...] / (1.0 / phys.zvir + qbot[...]))
        / ustar[...] ** 2
    )
    hol = npx.minimum(npx.abs(hol[...]), sett.bulkStabilityLimit) * npx.sign(hol[...])
    stable = 0.5 + 0.5 * npx.sign(hol[...])
    xsq = npx.maximum(
        npx.sqrt(npx.abs(1.0 - phys.bulkUnstableStabilityCoefficient * hol[...])), 1.0
    )
    xqq = npx.sqrt(xsq[...])
    psimh = -phys.bulkStableStabilityCoefficient * hol[...] * stable[...] + (
        1.0 - stable[...]
    ) * psimhu(phys, xqq[...])
    psixh = -phys.bulkStableStabilityCoefficient * hol[...] * stable[...] + (
        1.0 - stable[...]
    ) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / phys.karman * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = npx.sqrt(cdn(phys, u10n[...]))
    ren = phys.cesmNeutralMoisture
    rhn = (1.0 - stable[...]) * phys.cesmNeutralHeatUnstable + stable[
        ...
    ] * phys.cesmNeutralHeatStable

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / phys.karman * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / phys.karman * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / phys.karman * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # iterate to converge on Z/L, ustar, tstar and qstar

    # compute stability & evaluate all stability functions
    hol = (
        phys.karman
        * phys.gravity
        * zbot[...]
        * (tstar[...] / thbot[...] + qstar[...] / (1.0 / phys.zvir + qbot[...]))
        / ustar[...] ** 2
    )
    hol = npx.minimum(npx.abs(hol[...]), sett.bulkStabilityLimit) * npx.sign(hol[...])
    stable = 0.5 + 0.5 * npx.sign(hol[...])
    xsq = npx.maximum(
        npx.sqrt(npx.abs(1.0 - phys.bulkUnstableStabilityCoefficient * hol[...])), 1.0
    )
    xqq = npx.sqrt(xsq[...])
    psimh = -phys.bulkStableStabilityCoefficient * hol[...] * stable[...] + (
        1.0 - stable[...]
    ) * psimhu(phys, xqq[...])
    psixh = -phys.bulkStableStabilityCoefficient * hol[...] * stable[...] + (
        1.0 - stable[...]
    ) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / phys.karman * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = npx.sqrt(cdn(phys, u10n[...]))
    ren = phys.cesmNeutralMoisture
    rhn = (1.0 - stable[...]) * phys.cesmNeutralHeatUnstable + stable[
        ...
    ] * phys.cesmNeutralHeatStable

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / phys.karman * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / phys.karman * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / phys.karman * (alz[...] - psixh[...]))

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
    lat = phys.latvap * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -phys.stefBoltz * ts[...] ** 4 * mask[...]

    # water flux
    evap = lat[...] / phys.latvap * mask[...]

    # compute diagnositcs: 2m ref T & Q, 10m wind speed squared

    hol = hol[...] * sett.ztref / zbot[...]
    xsq = npx.maximum(
        1.0, npx.sqrt(npx.abs(1.0 - phys.bulkUnstableStabilityCoefficient * hol[...]))
    )
    xqq = npx.sqrt(xsq)
    psix2 = -phys.bulkStableStabilityCoefficient * hol[...] * stable[...] + (
        1.0 - stable[...]
    ) * psixhu(xqq[...])
    fac = (rh[...] / phys.karman) * (alz[...] + al2 - psixh[...] + psix2[...])
    tref = thbot[...] - delt[...] * fac[...]

    # pot. temp to temp correction
    tref = (tref[...] - phys.gamma_blk * sett.ztref) * mask[...]
    fac = (
        (re[...] / phys.karman) * (alz[...] + al2 - psixh[...] + psix2[...]) * mask[...]
    )
    qref = (qbot[...] - delq[...] * fac[...]) * mask[...]

    # 10m wind speed squared
    duu10n = u10n[...] * u10n[...] * mask[...]

    return (
        sen,
        lat,
        cast(Array, lwup),
        evap,
        taux,
        tauy,
        cast(Array, tref),
        cast(Array, qref),
        duu10n,
        ustar,
        tstar,
        qstar,
    )


@partial(jit, static_argnames=["sett", "phys"])
def flux_atmOcn_simple(
    sett: Settings,
    phys: PhysicalConstants,
    mask: MaskInput,
    ps: ArrayInput,
    qbot: ArrayInput,
    rbot: ArrayInput,
    ubot: ArrayInput,
    vbot: ArrayInput,
    tbot: ArrayInput,
    us: ArrayInput,
    vs: ArrayInput,
    ts: ArrayInput,
) -> HeatFluxes:
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

    vmag = npx.maximum(
        sett.umin_o,
        npx.sqrt((ubot[...] - us[...]) ** 2 + (vbot[...] - vs[...]) ** 2),
    )

    # long-wave radiation (IR)
    qir = -phys.stefBoltz * ts[...] ** 4 * mask[...]

    # sensible heat flux
    qh = (
        rbot[...]
        * phys.cpdair
        * phys.ch
        * vmag[...]
        * (tbot[...] - ts[...])
        * mask[...]
    )

    # latent heat flux
    qe = (
        -rbot[...]
        * phys.ce
        * phys.latvap
        * vmag[...]
        * (qsat_august_eqn(phys, ps, ts) - qbot[...])
        * mask[...]
    )

    return cast(Array, qir), cast(Array, qh), cast(Array, qe)
