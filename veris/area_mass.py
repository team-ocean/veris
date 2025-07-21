import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['sett'])
def AreaWS(vs, sett):
    """calculate sea ice cover fraction centered around velocity points"""

    AreaW = 0.5 * (vs.Area + jnp.roll(vs.Area, 1, 0))
    AreaS = 0.5 * (vs.Area + jnp.roll(vs.Area, 1, 1))

    return AreaW, AreaS

@partial(jax.jit, static_argnames=['sett'])
def SeaIceMass(vs, sett):
    """calculate mass of the ice-snow system centered around c-, u-, and v-points"""

    sett = state.settings

    SeaIceMassC = sett.rhoIce * vs.hIceMean + sett.rhoSnow * vs.hSnowMean
    SeaIceMassU = 0.5 * (SeaIceMassC + jnp.roll(SeaIceMassC, 1, 0))
    SeaIceMassV = 0.5 * (SeaIceMassC + jnp.roll(SeaIceMassC, 1, 1))

    return SeaIceMassC, SeaIceMassU, SeaIceMassV
