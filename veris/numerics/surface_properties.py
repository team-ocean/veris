"""Diagnose instantaneous per-ice-area albedo using native growth geometry.

The halo-inclusive horizontal state uses regularized concentration to recover
actual ice/snow thickness. The native dry/wet and hemispheric albedo law is
evaluated in every thickness category and averaged without concentration
weighting. Couplers apply that weighting on their source grid.
"""

from dataclasses import replace
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from veris.model.arrays import ArrayInput
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State, jit


class _ThicknessGeometry(NamedTuple):
    ice_mean: Array
    snow_mean: Array
    area: Array
    reciprocal_area: Array
    reciprocal_ice_thickness: Array
    ice_thickness: Array
    snow_thickness: Array


@jit
def _prepare_thickness_geometry(
    state: State, physical_constants: PhysicalConstants
) -> _ThicknessGeometry:
    no_ice = (state.hIceMean == 0) | (state.Area == 0)
    ice_mean = state.hIceMean * ~no_ice
    snow_mean = state.hSnowMean * ~no_ice
    area = state.Area * ~no_ice
    is_ice = ice_mean > 0
    area_squared = area**2 + physical_constants.Area_reg
    # The disabled regularizer still needs a finite inactive denominator.
    regularized_area = jnp.sqrt(jnp.where(area_squared == 0, 1, area_squared))
    reciprocal_area = 1 / regularized_area
    ice_thickness = jnp.where(is_ice, ice_mean * reciprocal_area, 0)
    reciprocal_ice_thickness = area / jnp.sqrt(
        ice_mean**2 + physical_constants.hIce_reg
    )
    snow_thickness = jnp.where(is_ice, snow_mean * reciprocal_area, 0)
    # Growth also solves ice initiation in currently ice-free wet cells.
    ice_thickness = jnp.where(
        state.iceMask != 0,
        jnp.maximum(ice_thickness, physical_constants.minActualIceThickness),
        0,
    )
    return _ThicknessGeometry(
        ice_mean,
        snow_mean,
        area,
        reciprocal_area,
        reciprocal_ice_thickness,
        ice_thickness,
        snow_thickness,
    )


@partial(jit, static_argnames=["configuration"])
def _category_factors(configuration: Configuration) -> Array:
    return jnp.asarray(
        [
            (2 * (index + 1) - 1) * configuration.recip_nITC
            for index in range(configuration.nITC)
        ],
        dtype=configuration.dtype,
    )


@jit
def _surface_albedo(
    temperature: ArrayInput,
    ice_thickness: ArrayInput,
    snow_thickness: ArrayInput,
    coriolis: ArrayInput,
    physical_constants: PhysicalConstants,
) -> Array:
    phys = physical_constants
    is_ice = ice_thickness > 0
    melt_temperature = phys.celsius2K + phys.wetAlbTemp
    ice_albedo = jnp.where(is_ice, phys.dryIceAlb, 0)
    snow_albedo = jnp.where(is_ice, phys.drySnowAlb, 0)
    wet = (ice_thickness > 0) & (temperature >= melt_temperature)
    ice_albedo = jnp.where(wet, phys.wetIceAlb, ice_albedo)
    snow_albedo = jnp.where(wet, phys.wetSnowAlb, snow_albedo)
    south = (ice_thickness > 0) & (coriolis < 0)
    ice_albedo = jnp.where(south, phys.dryIceAlb_south, ice_albedo)
    snow_albedo = jnp.where(south, phys.drySnowAlb_south, snow_albedo)
    wet_south = (ice_thickness > 0) & (coriolis < 0) & (temperature >= melt_temperature)
    ice_albedo = jnp.where(wet_south, phys.wetIceAlb_south, ice_albedo)
    snow_albedo = jnp.where(wet_south, phys.wetSnowAlb_south, snow_albedo)
    albedo = jnp.where(
        is_ice,
        ice_albedo + snow_thickness / phys.hCut * (snow_albedo - ice_albedo),
        0,
    )
    albedo = jnp.where(snow_thickness > phys.hCut, snow_albedo, albedo)
    return jnp.where(snow_thickness == 0, ice_albedo, albedo)


@partial(jit, static_argnames=["configuration"])
def diagnose_surface_albedo(
    state: State,
    configuration: Configuration,
    physical_constants: PhysicalConstants,
) -> Array:
    """Return current per-ice-area albedo on the halo-inclusive (x, y) grid.

    Current stored surface temperature and category snow thickness determine
    albedo; this is not an average of the preceding integration interval.
    Inactive cells return zero, and undefined inactive data have no derivative.
    """
    active = (state.iceMask != 0) & (state.Area > 0) & (state.hIceMean > 0)
    state = replace(
        state,
        **{
            name: jnp.where(active, getattr(state, name), 0)
            for name in ("hIceMean", "hSnowMean", "Area", "TSurf", "fCori")
        },
    )
    geometry = _prepare_thickness_geometry(state, physical_constants)
    factors = _category_factors(configuration)
    category_albedo = _surface_albedo(
        state.TSurf[..., None],
        geometry.ice_thickness[..., None] * factors,
        geometry.snow_thickness[..., None] * factors,
        state.fCori[..., None],
        physical_constants,
    )
    result = jnp.sum(category_albedo, axis=-1) * configuration.recip_nITC
    return jnp.where(active, result, 0)
