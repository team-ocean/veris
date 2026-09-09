"""Standalone JAX bulk heat-flux kernels, retaining the original MITgcm LANL equations.

Array inputs preserve the original shapes and units documented per function.
Settings and physical constants are supplied as separate frozen dataclasses.
"""

from functools import partial

import jax.numpy as npx
from jax.typing import ArrayLike

from veris._bulk_types import LANLFluxes
from veris._typing import jit
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants


@partial(jit, static_argnames=["sett", "phys"])
def bulkf_formula_lanl(
    sett: Settings,
    phys: PhysicalConstants,
    uw: ArrayLike,
    vw: ArrayLike,
    ta: ArrayLike,
    qa: ArrayLike,
    tsf: ArrayLike,
    ocn_mask: ArrayLike,
) -> LANLFluxes:
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
        ocn_mask (:obj:`ndarray`): 0=land, nonzero=ocean (not an area weight).
            Land diagnostics and input sensitivities are zero, including when
            land inputs contain missing values. MITgcm bulkf_forcing.F gates
            calls to the scalar LANL formula on nonzero maskC.

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

    # Evaluate inactive cells at a regular finite state before masking outputs.
    # Masking outputs alone would leave NaNs in the reverse differentiation pass.
    wet = ocn_mask != 0
    uw = npx.where(wet, uw, 1.0)
    vw = npx.where(wet, vw, 1.0)
    ta = npx.where(wet, ta, 275.0)
    qa = npx.where(wet, qa, 0.003)
    tsf = npx.where(wet, tsf, 280.0)

    # Compute turbulent surface fluxes
    ht = sett.ztref
    zref = sett.zref
    zice = phys.zzsice
    aln = npx.log(ht / zref)
    czol = zref * phys.karman * phys.gravity

    lath = npx.ones_like(ocn_mask) * phys.latvap

    # wind speed
    us = npx.sqrt(uw[...] * uw[...] + vw[...] * vw[...])
    usm = npx.maximum(us[...], sett.lanlMinWindSpeed)

    t0 = ta[...] * (1.0 + phys.zvir * qa[...])
    ssq = (
        phys.lanlSaturationHumidityScale
        * npx.exp(
            lath[...]
            * (
                phys.lanlSaturationExponentOffset
                - phys.lanlSaturationExponentTemperature / tsf[...]
            )
        )
        / phys.lanlReferencePressure
    )

    deltap = ta[...] - tsf[...] + phys.gamma_blk * ht
    delq = qa[...] - ssq[...]

    # initialize estimate exchange coefficients
    rdn = phys.karman / npx.log(zref / zice)
    rhn = rdn
    ren = rdn
    # calculate turbulent scales
    ustar = rdn * usm[...]
    tstar = rhn * deltap[...]
    qstar = ren * delq[...]

    # iteration with psi-functions to find transfer coefficients
    for _ in range(sett.lanlBulkIterations):
        huol = (
            czol
            / ustar[...] ** 2
            * (tstar[...] / t0 + qstar[...] / (1.0 / phys.zvir + qa[...]))
        )
        huol = npx.minimum(npx.abs(huol[...]), sett.bulkStabilityLimit) * npx.sign(
            huol[...]
        )
        stable = 0.5 + 0.5 * npx.sign(huol[...])
        xsq = npx.maximum(
            npx.sqrt(npx.abs(1.0 - phys.bulkUnstableStabilityCoefficient * huol[...])),
            1.0,
        )
        x = npx.sqrt(xsq[...])
        psimh = -phys.bulkStableStabilityCoefficient * huol[...] * stable[...] + (
            1.0 - stable[...]
        ) * (
            2.0 * npx.log(0.5 * (1.0 + x[...]))
            + 2.0 * npx.log(0.5 * (1.0 + xsq[...]))
            - 2.0 * npx.arctan(x[...])
            + npx.pi * 0.5
        )
        psixh = -phys.bulkStableStabilityCoefficient * huol[...] * stable[...] + (
            1.0 - stable[...]
        ) * (2.0 * npx.log(0.5 * (1.0 + xsq[...])))

        # update the transfer coefficients
        rd = rdn / (1.0 + rdn * (aln[...] - psimh[...]) / phys.karman)
        rh = rhn / (1.0 + rhn * (aln[...] - psixh[...]) / phys.karman)
        re = rh

        # update ustar, tstar, qstar using updated, shifted coefficients.
        ustar = rd[...] * usm[...]
        qstar = re[...] * delq[...]
        tstar = rh[...] * deltap[...]

    # tau = phys.rhoAir * ustar[...]**2
    # tau = tau * us[...] / usm[...]
    csha = phys.rhoAir * phys.cpdair * us[...] * rh[...] * rd[...]
    clha = phys.rhoAir * lath[...] * us[...] * re[...] * rd[...]

    fsha = csha[...] * deltap[...]
    flha = clha[...] * delq[...]
    evp = -flha[...] / lath[...]

    flwupa = phys.ocean_emissivity * phys.stefBoltz * tsf[...] ** 4
    dflwupdt = 4.0 * phys.ocean_emissivity * phys.stefBoltz * tsf[...] ** 3

    devdt = (
        clha[...]
        * ssq[...]
        * phys.lanlSaturationExponentTemperature
        / (tsf[...] * tsf[...])
    )
    dflhdt = -lath[...] * devdt[...]
    dfshdt = -csha[...]

    # total derivative with respect to surface temperature
    df0dt = -dflwupdt[...] + dfshdt[...] + dflhdt[...]

    #  wind stress at center points
    bulkf_cdn = (
        phys.neutralDragInverseWind / usm[...]
        + phys.neutralDragConstant
        + phys.neutralDragLinearWind * usm[...]
    )
    ust = phys.rhoAir * bulkf_cdn * us[...] * uw[...]
    vst = phys.rhoAir * bulkf_cdn * us[...] * vw[...]

    return (
        npx.where(wet, flwupa, 0.0),
        npx.where(wet, flha, 0.0),
        npx.where(wet, fsha, 0.0),
        npx.where(wet, df0dt, 0.0),
        npx.where(wet, ust, 0.0),
        npx.where(wet, vst, 0.0),
        npx.where(wet, evp, 0.0),
        npx.where(wet, ssq, 0.0),
        npx.where(wet, devdt, 0.0),
    )
