"""Standalone dynamics and growth on a periodic Cartesian sea with an island.

The small example follows the jax_halo_exchange reference integration sequence.
It uses prescribed ocean/atmospheric fields, not an evolving ocean or geographic
coastline. Fields are (nx + 4, ny + 4) JAX arrays with two periodic halo cells.
The host driver selects serial halos; state is an immutable namedtuple PyTree.
Five EVP substeps keep this example small; they are not a convergence criterion.
"""

import jax
import jax.numpy as jnp
import numpy as np

from veris.settings import settings
from veris.state import Settings, State


def initialize(
    nx: int = 8, ny: int = 12, wind: float = 5.0, air_temperature: float = 260.0
) -> tuple[State, Settings]:
    """Return an artificial masked state and settings for 600-second steps.

    Call in a fresh process before importing distributed halo consumers.
    The central two-by-two island blocks face transport on both sides. Grid
    spacing is 8 km; initial mean thickness/concentration are 1 m and 0.8.
    Atmosphere is saturated at its specified temperature, with blackbody
    downward longwave radiation and no sunlight or precipitation.
    """
    if nx < 4 or ny < 4:
        raise ValueError("grid dimensions must be at least four interior cells")
    settings["use_sharding"] = False
    sett = Settings(**settings)._replace(
        deltatTherm=600,
        recip_deltatTherm=1 / 600,
        deltatDyn=600,
        recip_deltatDyn=1 / 600,
        nEVPsteps=5,
    )
    ones = jnp.ones((nx + 4, ny + 4))
    fields = {name: jnp.zeros_like(ones) for name in State._fields}
    interior = np.ones((nx, ny))
    interior[nx // 2 - 1 : nx // 2 + 1, ny // 2 - 1 : ny // 2 + 1] = 0
    mask = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    west = mask * jnp.roll(mask, 1, axis=0)
    south = mask * jnp.roll(mask, 1, axis=1)
    fields.update(
        iceMask=mask,
        maskInC=mask,
        iceMaskU=west,
        maskInU=west,
        iceMaskV=south,
        maskInV=south,
    )
    for name in ("dxC", "dyC", "dxG", "dyG", "dxU", "dyU", "dxV", "dyV"):
        fields[name] = 8000 * ones
        fields["recip_" + name] = ones / 8000
    for name in ("rA", "rAu", "rAv", "rAz"):
        fields[name] = 8000**2 * ones
        fields["recip_" + name] = ones / 8000**2
    temperature = sett.celsius2K + sett.tempFrz
    vapor = 10 ** (12.537 - 2663.5 / air_temperature)
    fields.update(
        hIceMean=mask,
        hSnowMean=0.05 * mask,
        Area=0.8 * mask,
        TSurf=air_temperature * ones,
        SeaIceLoad=(sett.rhoIce + 0.05 * sett.rhoSnow) * mask,
        recip_hIceMean=1 / jnp.sqrt(mask**2 + sett.hIce_reg),
        R_low=-100 * ones,
        fCori=1e-4 * ones,
        theta=temperature * ones,
        ocSalt=34.7 * ones,
        uWind=wind * ones,
        wSpeed=abs(wind) * ones,
        ATemp=air_temperature * ones,
        LWdown=sett.stefBoltz * air_temperature**4 * ones,
        aqh=0.622 * vapor / (100000 - 0.378 * vapor) * ones,
    )
    return State._make(fields[name] for name in State._fields), sett


def step(vs: State, sett: Settings, cooling: float = 100.0) -> State:
    """Advance dynamics, transport, cleanup, and growth with prescribed forcing.

    Cooling is the upward open-water net heat flux in W/m². It is restored on
    every call because Growth returns ocean-coupling Qnet/Qsw in those fields.
    This example uses the serial halo backend selected by initialize().
    """
    from veris.advection import Advection
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.clean_up import clean_up_advection, ridging
    from veris.dynamics_routines import SeaIceStrength
    from veris.dynsolver import IceVelocities, WindForcingXY
    from veris.fill_overlap import fill_overlap
    from veris.growth import Growth
    from veris.ocean_stress import OceanStressUV

    def assign(state: State, names: str, values: tuple[jax.Array, ...]) -> State:
        return state._replace(**dict(zip(names.split(), values, strict=True)))

    vs = vs._replace(Qnet=jnp.full_like(vs.Qnet, cooling), Qsw=jnp.zeros_like(vs.Qsw))
    vs = assign(vs, "SeaIceMassC SeaIceMassU SeaIceMassV", SeaIceMass(vs, sett))
    vs = assign(vs, "AreaW AreaS", AreaWS(vs, sett))
    vs = assign(vs, "WindForcingX WindForcingY", WindForcingXY(vs, sett))
    vs = vs._replace(SeaIceStrength=SeaIceStrength(vs, sett))
    vs = assign(vs, "uIce vIce sigma1 sigma2 sigma12", IceVelocities(vs, sett))
    vs = assign(vs, "OceanStressU OceanStressV", OceanStressUV(vs, sett))
    vs = assign(vs, "hIceMean hSnowMean Area", Advection(vs, sett))
    vs = assign(
        vs,
        "hIceMean hSnowMean Area TSurf os_hIceMean os_hSnowMean",
        clean_up_advection(vs, sett),
    )
    vs = vs._replace(Area=ridging(vs, sett))
    vs = assign(
        vs,
        "hIceMean hSnowMean Area TSurf EmPmR forc_salt_surface Qsw Qnet "
        "SeaIceLoad IcePenetSW recip_hIceMean",
        Growth(vs, sett),
    )
    return jax.tree.map(fill_overlap, vs)
