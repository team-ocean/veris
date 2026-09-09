"""Concrete immutable PyTrees used by the standalone integration example.

Kernel arguments use structural domain protocols, so other PyTree containers
remain supported. These schemas preserve the order of the public registries.
Physical constants accept floats even when a default happens to be an integer.
Horizontal state fields include the two-cell periodic halos.
"""

from dataclasses import dataclass

import jax
from jax import Array

from veris.configuration import Settings

__all__ = ["Settings", "State"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class State:
    """Calculation fields on the halo-inclusive C grid, in registry order.

    Construct through :func:`veris.initialization.initialize` to allocate
    defaults. All fields are dynamic array leaves for JAX differentiation;
    settings, physical constants and output-only diagnostics live separately.
    Use :func:`dataclasses.replace` for immutable numerical updates.
    """

    hIceMean: Array
    hSnowMean: Array
    Area: Array
    TSurf: Array
    SeaIceMassC: Array
    SeaIceMassU: Array
    SeaIceMassV: Array
    SeaIceStrength: Array
    os_hIceMean: Array
    os_hSnowMean: Array
    AreaW: Array
    AreaS: Array
    uIce: Array
    vIce: Array
    sigma1: Array
    sigma2: Array
    sigma12: Array
    WindForcingX: Array
    WindForcingY: Array
    recip_hIceMean: Array
    SeaIceLoad: Array
    uOcean: Array
    vOcean: Array
    theta: Array
    ocSalt: Array
    Qnet: Array
    R_low: Array
    ssh_an: Array
    Qsw: Array
    uWind: Array
    vWind: Array
    wSpeed: Array
    surfPress: Array
    SWdown: Array
    LWdown: Array
    ATemp: Array
    aqh: Array
    precip: Array
    snowfall: Array
    evap: Array
    runoff: Array
    maskInC: Array
    maskInU: Array
    maskInV: Array
    iceMask: Array
    iceMaskU: Array
    iceMaskV: Array
    k1AtC: Array
    k2AtC: Array
    k1AtZ: Array
    k2AtZ: Array
    Fu: Array
    Fv: Array
    fCori: Array
    dxG: Array
    dyG: Array
    dxU: Array
    dyU: Array
    dxV: Array
    dyV: Array
    recip_dxC: Array
    recip_dyC: Array
    recip_dxU: Array
    recip_dyU: Array
    recip_dxV: Array
    recip_dyV: Array
    rAz: Array
    recip_rA: Array
    recip_rAu: Array
    recip_rAv: Array
