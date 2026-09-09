"""Redistribute thin ice and bound transported concentration and thickness."""

from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from jax import Array

from veris._typing import jit
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.state import State


@partial(jit, static_argnames=["sett", "phys"])
def clean_up_advection(
    vs: State, sett: Settings, phys: PhysicalConstants
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """clean up overshoots and other pathological cases after advection"""

    # case 1: negative values
    # calculate overshoots of ice and snow thickness
    os_hIceMean = jnp.maximum(-vs.hIceMean, 0)
    os_hSnowMean = jnp.maximum(-vs.hSnowMean, 0)

    # cut off thicknesses and area at zero
    hIceMean = jnp.maximum(vs.hIceMean, 0)
    hSnowMean = jnp.maximum(vs.hSnowMean, 0)
    Area = jnp.maximum(vs.Area, 0)

    # case 2: very thin ice
    # set thicknesses to zero if the ice thickness is very small
    thinIce = hIceMean <= sett.hIce_min
    hIceMean *= ~thinIce
    hSnowMean *= ~thinIce
    TSurf = jnp.where(thinIce, phys.celsius2K, vs.TSurf)

    # case 3: area but no ice and snow
    # set area to zero if no ice or snow is present
    Area = jnp.where((hIceMean == 0) & (hSnowMean == 0), 0, Area)

    # case 4: very small area
    # introduce lower boundary for the area (if ice or snow is present)
    Area = jnp.where(
        (hIceMean > 0) | (hSnowMean > 0), jnp.maximum(Area, sett.Area_min), Area
    )

    return hIceMean, hSnowMean, Area, TSurf, os_hIceMean, os_hSnowMean


@partial(jit, static_argnames=["sett", "phys"])
def ridging(vs: State, sett: Settings, phys: PhysicalConstants) -> Array:
    """cut off ice cover fraction at 1 after advection to account for ridging"""
    Area = jnp.minimum(vs.Area, 1)

    return Area
