from functools import partial

import jax
import jax.numpy as jnp

from veris.fill_overlap import fill_overlap

# in this routine, the thermodynamic time step is used instead of the dynamic one.
# this has historical reasons as with lower resolutions, the dynamics change much
# slower than the thermodynamics (thermodynamics have a daily cycle). calculating
# the ice velocity as often as the thermodynamics was unnecessarily expensive but
# the advection is still done with the faster thermodynamic timestep as the ice
# thickness changes inbetween dynamics timesteps.


@partial(jax.jit, static_argnames=["sett"])
def Advection(vs, sett):
    """retrieve changes in sea ice fields"""

    hIceMean = calc_Advection(vs, sett, vs.hIceMean)
    hSnowMean = calc_Advection(vs, sett, vs.hSnowMean)
    Area = calc_Advection(vs, sett, vs.Area)

    return hIceMean, hSnowMean, Area


@partial(jax.jit, static_argnames=["sett"])
def calc_Advection(vs, sett, field):
    """calculate change in sea ice field due to advection"""

    # retrieve cell faces
    xA = vs.dyG * vs.iceMaskU
    yA = vs.dxG * vs.iceMaskV

    # calculate ice transport
    uTrans = vs.uIce * xA
    vTrans = vs.vIce * yA

    # make local copy of field prior to advective changes
    fieldLoc = field

    # calculate zonal advective fluxes
    ZonalFlux = calc_ZonalFlux(vs, sett, fieldLoc, uTrans)

    # update field according to zonal fluxes
    if sett.extensiveFld:
        fieldLoc = fieldLoc - sett.deltatTherm * vs.maskInC * vs.recip_rA * (
            jnp.roll(ZonalFlux, -1, 0) - ZonalFlux
        )
    else:
        fieldLoc = (
            fieldLoc
            - sett.deltatTherm
            * vs.maskInC
            * vs.recip_rA
            * vs.recip_hIceMean
            * (
                (jnp.roll(ZonalFlux, -1, 0) - ZonalFlux)
                - (jnp.roll(uTrans, -1, 0) - uTrans) * field
            )
        )

    # calculate meridional advective fluxes
    MeridionalFlux = calc_MeridionalFlux(vs, sett, fieldLoc, vTrans)

    # update field according to meridional fluxes
    if sett.extensiveFld:
        fieldLoc = fieldLoc - sett.deltatTherm * vs.maskInC * vs.recip_rA * (
            jnp.roll(MeridionalFlux, -1, 1) - MeridionalFlux
        )
    else:
        fieldLoc = (
            fieldLoc
            - sett.deltatTherm
            * vs.maskInC
            * vs.recip_rA
            * vs.recip_hIceMean
            * (
                (jnp.roll(MeridionalFlux, -1, 1) - MeridionalFlux)
                - (jnp.roll(vTrans, -1, 1) - vTrans) * fieldLoc
            )
        )

    # apply mask
    fieldLoc = fieldLoc * vs.iceMask

    return fieldLoc


@partial(jax.jit, static_argnames=["sett"])
def calc_ZonalFlux(vs, sett, field, uTrans):
    """calculate the zonal advective flux using the second order flux limiter method"""

    maskLocW = vs.iceMaskU * vs.maskInU

    # CFL number of zonal flow
    uCFL = jnp.abs(vs.uIce * sett.deltatTherm * vs.recip_dxC)

    # calculate slope ratio Cr
    Rjp = (field[3:, :] - field[2:-1, :]) * maskLocW[3:, :]
    Rj = (field[2:-1, :] - field[1:-2, :]) * maskLocW[2:-1, :]
    Rjm = (field[1:-2, :] - field[:-3, :]) * maskLocW[1:-2, :]

    Cr = jnp.where(uTrans[2:-1, :] > 0, Rjm, Rjp)
    Cr = jnp.where(
        jnp.abs(Rj) * sett.CrMax > jnp.abs(Cr),
        Cr / Rj,
        jnp.sign(Cr) * sett.CrMax * jnp.sign(Rj),
    )
    Cr = limiter(Cr)

    # zonal advective flux for the given field
    ZonalFlux = jnp.zeros(vs.iceMask.shape)
    ZonalFlux = ZonalFlux.at[2:-1, :].set(
        uTrans[2:-1, :] * (field[2:-1, :] + field[1:-2, :]) * 0.5
        - jnp.abs(uTrans[2:-1, :]) * ((1 - Cr) + uCFL[2:-1, :] * Cr) * Rj * 0.5,
    )
    ZonalFlux = fill_overlap(ZonalFlux)

    return ZonalFlux


@partial(jax.jit, static_argnames=["sett"])
def calc_MeridionalFlux(vs, sett, field, vTrans):
    """calculate the meridional advective flux using the second order flux limiter method"""

    maskLocS = vs.iceMaskV * vs.maskInV

    # CFL number of meridional flow
    vCFL = jnp.abs(vs.vIce * sett.deltatTherm * vs.recip_dyC)

    # calculate slope ratio Cr
    Rjp = (field[:, 3:] - field[:, 2:-1]) * maskLocS[:, 3:]
    Rj = (field[:, 2:-1] - field[:, 1:-2]) * maskLocS[:, 2:-1]
    Rjm = (field[:, 1:-2] - field[:, :-3]) * maskLocS[:, 1:-2]

    Cr = jnp.where(vTrans[:, 2:-1] > 0, Rjm, Rjp)
    Cr = jnp.where(
        jnp.abs(Rj) * sett.CrMax > jnp.abs(Cr),
        Cr / Rj,
        jnp.sign(Cr) * sett.CrMax * jnp.sign(Rj),
    )
    Cr = limiter(Cr)

    # meridional advective flux for the given field
    MeridionalFlux = jnp.zeros(vs.iceMask.shape)
    MeridionalFlux = MeridionalFlux.at[:, 2:-1].set(
        vTrans[:, 2:-1] * (field[:, 2:-1] + field[:, 1:-2]) * 0.5
        - jnp.abs(vTrans[:, 2:-1]) * ((1 - Cr) + vCFL[:, 2:-1] * Cr) * Rj * 0.5,
    )
    MeridionalFlux = fill_overlap(MeridionalFlux)

    return MeridionalFlux


@partial(jax.jit)
def limiter(Cr):
    # return 0       (upwind)
    # return 1       (Lax-Wendroff)
    # return np.max((0, np.min((1, Cr))))    (Min-Mod)
    return jnp.maximum(0, jnp.maximum(jnp.minimum(1, 2 * Cr), jnp.minimum(2, Cr)))
