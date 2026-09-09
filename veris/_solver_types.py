"""Structural interfaces for wind forcing and sea-ice momentum solvers.

Each read-only protocol composes exactly the fields needed by a solver and its
callees. Geometry and forcing fields are horizontal JAX arrays with halos;
settings are static scalar values. EVP iteration counts are integers.
"""

from typing import Protocol

from jax import Array

from veris._dynamics_types import (
    BasalDragSettings,
    BasalDragState,
    FaceMaskState,
    FreeDriftSettings,
    FreeDriftState,
    OceanDragSettings,
    OceanDragState,
    SideDragSettings,
    SideDragState,
    StrainSettings,
    StrainState,
    StressDivergenceState,
    ViscositySettings,
    ViscosityState,
)
from veris._typing import StaticSettings


class WindStressState(FaceMaskState, Protocol):
    """Wind and ice velocities, Coriolis parameter and face masks."""

    @property
    def uWind(self) -> Array: ...

    @property
    def vWind(self) -> Array: ...

    @property
    def uIce(self) -> Array: ...

    @property
    def vIce(self) -> Array: ...

    @property
    def fCori(self) -> Array: ...


class WindStressSettings(StaticSettings, Protocol):
    """Relative wind selection and minimum wind speed."""

    @property
    def useRelativeWind(self) -> bool: ...

    @property
    def wSpeedMin(self) -> float: ...


class WindForcingState(WindStressState, Protocol):
    """Wind stress inputs plus surface tilt, pressure and ice load."""

    @property
    def AreaW(self) -> Array: ...

    @property
    def AreaS(self) -> Array: ...

    @property
    def ssh_an(self) -> Array: ...

    @property
    def surfPress(self) -> Array: ...

    @property
    def SeaIceLoad(self) -> Array: ...

    @property
    def SeaIceMassU(self) -> Array: ...

    @property
    def SeaIceMassV(self) -> Array: ...

    @property
    def recip_dxC(self) -> Array: ...

    @property
    def recip_dyC(self) -> Array: ...


class WindForcingSettings(WindStressSettings, StaticSettings, Protocol):
    """Freshwater-load selection and surface-load multiplier."""

    @property
    def useRealFreshWaterFlux(self) -> bool: ...

    @property
    def seaIceLoadFac(self) -> float: ...


class EVPState(
    BasalDragState,
    OceanDragState,
    SideDragState,
    StrainState,
    ViscosityState,
    StressDivergenceState,
    Protocol,
):
    """Geometry and constitutive inputs for iterative ice momentum."""

    @property
    def uIce(self) -> Array: ...

    @property
    def vIce(self) -> Array: ...

    @property
    def sigma1(self) -> Array: ...

    @property
    def sigma2(self) -> Array: ...

    @property
    def sigma12(self) -> Array: ...

    @property
    def SeaIceMassC(self) -> Array: ...

    @property
    def WindForcingX(self) -> Array: ...

    @property
    def WindForcingY(self) -> Array: ...


class EVPSettings(
    OceanDragSettings,
    BasalDragSettings,
    SideDragSettings,
    StrainSettings,
    ViscositySettings,
    StaticSettings,
    Protocol,
):
    """Constitutive parameters and static controls for EVP subcycling."""

    @property
    def computeEvpResidual(self) -> bool: ...

    @property
    def useAdaptiveEVP(self) -> bool: ...

    @property
    def aEVPalphaMin(self) -> float: ...

    @property
    def recip_deltatDyn(self) -> float: ...

    @property
    def deltatDyn(self) -> float: ...

    @property
    def aEvpCoeff(self) -> float: ...

    @property
    def evpAlpha(self) -> float: ...

    @property
    def evpBeta(self) -> float: ...

    @property
    def nEVPsteps(self) -> int: ...

    @property
    def aEVPmassMin(self) -> float: ...

    @property
    def aEVPcStar(self) -> float: ...

    @property
    def evpStressRelaxation(self) -> float: ...

    @property
    def evpShearRelaxation(self) -> float: ...

    @property
    def printEvpResidual(self) -> bool: ...

    @property
    def use_sharding(self) -> bool: ...


class IceVelocityState(EVPState, FreeDriftState, Protocol):
    """Inputs shared by the free-drift and EVP dispatcher branches."""


class IceVelocitySettings(EVPSettings, FreeDriftSettings, StaticSettings, Protocol):
    """Momentum parameters and static solver selection."""

    @property
    def useFreedrift(self) -> bool: ...

    @property
    def useEVP(self) -> bool: ...


# fori_loop promotes the initial scalar factor to a scalar JAX tracer.
type EVPCarry = tuple[
    EVPState,
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
