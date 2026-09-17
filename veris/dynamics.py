"""Sea-ice dynamics and transport composed from the core numerical kernels.

Compose the existing Hibler/Kimmritz momentum kernels, ocean stress, directional
advection, cleanup and ridging in reference order. Inputs and returned State and
stress arrays retain their halo-inclusive (x, y) storage. Callers own atmospheric
forcing, optional thermodynamics and the final State/Diagnostics halo refresh.
Under sharding this stage must run inside the caller's shard_map.
"""

from dataclasses import replace

import jax

from veris._typing import State
from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants


def dynamics_transport(
    vs: State, conf: Configuration, phys: PhysicalConstants
) -> tuple[State, jax.Array, jax.Array]:
    """Advance momentum and transport, retaining pre-transport ocean stresses."""
    from veris.advection import Advection
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.clean_up import clean_up_advection, ridging
    from veris.dynamics_routines import SeaIceStrength
    from veris.dynsolver import IceVelocities, WindForcingXY
    from veris.ocean_stress import OceanStressUV

    def assign(state: State, names: str, values: tuple[jax.Array, ...]) -> State:
        return replace(state, **dict(zip(names.split(), values, strict=True)))

    vs = assign(vs, "SeaIceMassC SeaIceMassU SeaIceMassV", SeaIceMass(vs, conf, phys))
    vs = assign(vs, "AreaW AreaS", AreaWS(vs, conf, phys))
    vs = assign(vs, "WindForcingX WindForcingY", WindForcingXY(vs, conf, phys))
    vs = replace(vs, SeaIceStrength=SeaIceStrength(vs, conf, phys))
    vs = assign(
        vs,
        "uIce vIce sigma1 sigma2 sigma12",
        IceVelocities(
            vs, conf, phys, axis_names=("x", "y") if conf.use_sharding else ()
        ),
    )
    stress_u, stress_v = OceanStressUV(vs, conf, phys)
    vs = assign(vs, "hIceMean hSnowMean Area", Advection(vs, conf, phys))
    vs = assign(
        vs,
        "hIceMean hSnowMean Area TSurf os_hIceMean os_hSnowMean",
        clean_up_advection(vs, conf, phys),
    )
    vs = replace(vs, Area=ridging(vs, conf, phys))
    return vs, stress_u, stress_v
