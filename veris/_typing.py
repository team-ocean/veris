"""Shared Veris schemas, array contracts and typed JAX compiled callables.

State is an array-only PyTree; OceanGeometry describes host input.
Configuration and PhysicalConstants are defined beside their respective registries.
Artificial experiment types belong to their setup module.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ParamSpec, Protocol, TypeVar, cast

import jax
import numpy as np
from jax import Array
from jax.stages import Lowered, Traced
from jax.typing import ArrayLike
from numpy.typing import NDArray

type ArrayInput = Array | NDArray[np.number]
"""Indexable numerical input accepted by JAX at a compiled-kernel boundary."""


type MaskInput = ArrayInput | NDArray[np.bool_]
"""Indexable masks and halo fields may use NumPy boolean arrays as well."""


# JAX 0.11.1's JitWrapped signature erases parameter types. This boundary
# preserves them without wrapping the compiled call or altering its execution.
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)
R = TypeVar("R")


class Kernel(Protocol[P, R_co]):
    """A compiled callable with its original signature and lowering interface."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> Lowered: ...

    def trace(self, *args: P.args, **kwargs: P.kwargs) -> Traced: ...

    def eval_shape(self, *args: P.args, **kwargs: P.kwargs) -> object:
        """Return the corresponding shape PyTree, whose container is caller-defined."""
        ...

    def clear_cache(self) -> None: ...

    @property
    def __wrapped__(self) -> Callable[P, R_co]: ...


def jit(
    function: Callable[P, R],
    *,
    static_argnames: str | Sequence[str] | None = None,
) -> Kernel[P, R]:
    """Compile with JAX while retaining the input callable's static signature."""
    return cast(Kernel[P, R], jax.jit(function, static_argnames=static_argnames))


@dataclass(frozen=True)
class OceanGeometry:
    """Read-only ocean grid inputs, independent of any ocean-model container.

    ``maskT``, ``maskU`` and ``maskV`` are volume masks; ``dxt`` and ``dxu``
    are x-spacing vectors, and ``dyt`` and ``dyu`` are y-spacing vectors (m).
    ``ht`` is depth (m), ``coriolis_t`` is Coriolis frequency (s-1), and the
    three horizontal cell-area arrays are in m2. Shapes and finite positive
    metrics are checked by the host adapter before any reciprocal is computed.
    """

    maskT: ArrayInput
    maskU: ArrayInput
    maskV: ArrayInput
    ht: ArrayInput
    coriolis_t: ArrayInput
    dxt: ArrayInput
    dxu: ArrayInput
    dyt: ArrayInput
    dyu: ArrayInput
    area_t: ArrayInput
    area_u: ArrayInput
    area_v: ArrayInput


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


HeatFluxes = tuple[Array, Array, Array]
CESMFluxes = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]
LANLFluxes = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]


type SurfaceFluxResult = tuple[Array, Array, Array, Array, Array]
type GrowthResult = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]


type EVPCarry = tuple[
    State,
    Array,  # uIce
    Array,  # vIce
    Array,  # uIceNm1
    Array,  # vIceNm1
    Array,  # sigma1
    Array,  # sigma2
    Array,  # sigma12
    Array,  # denom1
    Array,  # denom2
    float | Array,  # EVPcFac before/during tracing
    Array,  # evpAlphaC
    Array,  # evpAlphaZ
    Array,  # evpBetaU
    Array,  # evpBetaV
    Array,  # resSig
    Array,  # resU
]


SumInput = TypeVar("SumInput", bound=ArrayLike)
