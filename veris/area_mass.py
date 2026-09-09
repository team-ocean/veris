"""Ice concentration and ice/snow mass at cell centers and velocity points."""

from collections.abc import Hashable
from functools import partial

import jax.numpy as jnp
from jax import Array

from veris._typing import AreaState, MassSettings, ThicknessState, jit
from veris.physical_constants import PhysicalConstants


@partial(jit, static_argnames=["sett", "phys"])
def AreaWS(
    vs: AreaState, sett: Hashable, phys: PhysicalConstants
) -> tuple[Array, Array]:
    """calculate sea ice cover fraction centered around velocity points"""

    AreaW = 0.5 * (vs.Area + jnp.roll(vs.Area, 1, 0))
    AreaS = 0.5 * (vs.Area + jnp.roll(vs.Area, 1, 1))

    return AreaW, AreaS


@partial(jit, static_argnames=["sett", "phys"])
def SeaIceMass(
    vs: ThicknessState, sett: MassSettings, phys: PhysicalConstants
) -> tuple[Array, Array, Array]:
    """calculate mass of the ice-snow system centered around c-, u-, and v-points"""

    SeaIceMassC = phys.rhoIce * vs.hIceMean + phys.rhoSnow * vs.hSnowMean
    SeaIceMassU = 0.5 * (SeaIceMassC + jnp.roll(SeaIceMassC, 1, 0))
    SeaIceMassV = 0.5 * (SeaIceMassC + jnp.roll(SeaIceMassC, 1, 1))

    return SeaIceMassC, SeaIceMassU, SeaIceMassV
