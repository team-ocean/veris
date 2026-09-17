"""Flux-limited directional transport of horizontal sea-ice fields."""

from __future__ import annotations

from functools import partial
from typing import cast

import jax.numpy as jnp
from jax import Array

from veris._typing import ArrayInput, State, jit
from veris.configuration import Configuration
from veris.fill_overlap import fill_overlap
from veris.physical_constants import PhysicalConstants

# in this routine, the thermodynamic time step is used instead of the dynamic one.
# this has historical reasons as with lower resolutions, the dynamics change much
# slower than the thermodynamics (thermodynamics have a daily cycle). calculating
# the ice velocity as often as the thermodynamics was unnecessarily expensive but
# the advection is still done with the faster thermodynamic timestep as the ice
# thickness changes inbetween dynamics timesteps.


@partial(jit, static_argnames=["conf", "phys"])
def Advection(
    vs: State, conf: Configuration, phys: PhysicalConstants
) -> tuple[Array, Array, Array]:
    """retrieve changes in sea ice fields"""

    hIceMean = calc_Advection(vs, conf, phys, vs.hIceMean)
    hSnowMean = calc_Advection(vs, conf, phys, vs.hSnowMean)
    Area = calc_Advection(vs, conf, phys, vs.Area)

    return hIceMean, hSnowMean, Area


@partial(jit, static_argnames=["conf", "phys"])
def calc_Advection(
    vs: State,
    conf: Configuration,
    phys: PhysicalConstants,
    field: ArrayInput,
) -> Array:
    """calculate change in sea ice field due to advection"""

    # retrieve cell faces
    xA = vs.dyG * vs.iceMaskU
    yA = vs.dxG * vs.iceMaskV

    # calculate ice transport
    uTrans = vs.uIce * xA
    vTrans = vs.vIce * yA
    if not conf.enable_cyclic_y:
        vTrans = fill_overlap(vTrans, conf, boundary="normal")

    fieldLoc = field
    for axis, transport, flux_kernel in (
        (0, uTrans, calc_ZonalFlux),
        (1, vTrans, calc_MeridionalFlux),
    ):
        flux = flux_kernel(vs, conf, phys, fieldLoc, transport)
        divergence = jnp.roll(flux, -1, axis) - flux
        if conf.extensiveFld:
            fieldLoc = (
                fieldLoc - conf.deltatTherm * vs.maskInC * vs.recip_rA * divergence
            )
        else:
            # Each directional correction uses the field entering that sweep.
            fieldLoc = (
                fieldLoc
                - conf.deltatTherm
                * vs.maskInC
                * vs.recip_rA
                * vs.recip_hIceMean
                * (divergence - (jnp.roll(transport, -1, axis) - transport) * fieldLoc)
            )

    # apply mask
    fieldLoc = fieldLoc * vs.iceMask

    # JIT converts NumPy field inputs to JAX tracers before these sweeps.
    return cast(Array, fieldLoc)


@partial(jit, static_argnames=["conf", "phys"])
def calc_ZonalFlux(
    vs: State,
    conf: Configuration,
    phys: PhysicalConstants,
    field: ArrayInput,
    uTrans: ArrayInput,
) -> Array:
    """calculate the zonal advective flux using the second order flux limiter method"""

    return _calc_flux(vs, conf, field, uTrans, axis=0)


@partial(jit, static_argnames=["conf", "phys"])
def calc_MeridionalFlux(
    vs: State,
    conf: Configuration,
    phys: PhysicalConstants,
    field: ArrayInput,
    vTrans: ArrayInput,
) -> Array:
    """calculate the meridional advective flux using the second order flux limiter method"""

    return _calc_flux(vs, conf, field, vTrans, axis=1)


def _calc_flux(
    vs: State,
    conf: Configuration,
    field: ArrayInput,
    transport: ArrayInput,
    *,
    axis: int,
) -> Array:
    """Apply the shared second-order limiter stencil along one C-grid axis.

    Keep storage orientation unchanged for rectangular and sharded grids. The
    meridional normal-face policy closes global walls; zonal flux extends edges.
    """
    if axis == 0:
        mask = vs.iceMaskU * vs.maskInU
        cfl = jnp.abs(vs.uIce * conf.deltatTherm * vs.recip_dxC)
    else:
        mask = vs.iceMaskV * vs.maskInV
        cfl = jnp.abs(vs.vIce * conf.deltatTherm * vs.recip_dyC)
    following, current, previous, before_previous = (
        tuple(window if dimension == axis else slice(None) for dimension in range(2))
        for window in (slice(3, None), slice(2, -1), slice(1, -2), slice(None, -3))
    )
    slope_next = (field[following] - field[current]) * mask[following]
    slope = (field[current] - field[previous]) * mask[current]
    slope_previous = (field[previous] - field[before_previous]) * mask[previous]
    upstream = jnp.where(transport[current] > 0, slope_previous, slope_next)
    uncapped = jnp.abs(slope) * conf.CrMax > jnp.abs(upstream)
    ratio = jnp.where(
        uncapped,
        upstream / jnp.where(uncapped, slope, 1.0),
        jnp.sign(upstream) * conf.CrMax * jnp.sign(slope),
    )
    limited = limiter(ratio)
    flux = (
        jnp.zeros_like(vs.iceMask)
        .at[current]
        .set(
            transport[current] * (field[current] + field[previous]) * 0.5
            - jnp.abs(transport[current])
            * ((1 - limited) + cfl[current] * limited)
            * slope
            * 0.5,
        )
    )
    return fill_overlap(flux, conf, boundary="edge" if axis == 0 else "normal")


@partial(jit)
def limiter(Cr: ArrayInput | float) -> Array:
    """Apply the Superbee slope limiter to a scalar or horizontal slope ratio."""
    # return 0       (upwind)
    # return 1       (Lax-Wendroff)
    # return np.max((0, np.min((1, Cr))))    (Min-Mod)
    return jnp.maximum(0, jnp.maximum(jnp.minimum(1, 2 * Cr), jnp.minimum(2, Cr)))
