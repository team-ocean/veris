"""Structural fields and constants for ice/snow thermodynamics.

Surface flux inputs are horizontal JAX arrays; Growth also consumes transported
ice/snow means and prescribed freshwater/ocean heat forcing. GrowthSettings
includes solve4temp's numerical controls because each thickness category calls
that solver. Physical coefficients are passed in a separate PhysicalConstants. nITC is an integer static loop/shape count. Properties preserve custom
immutable PyTrees, with no additional runtime state or numerical validation.
"""

from typing import Protocol

from jax import Array

from veris._typing import IceThermodynamicState, MaskState, MassSettings, StaticSettings

type SurfaceFluxResult = tuple[Array, Array, Array, Array, Array]
type GrowthResult = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]


class SurfaceState(Protocol):
    """Prescribed atmospheric forcing and Coriolis sign at cell centers."""

    @property
    def ATemp(self) -> Array: ...

    @property
    def LWdown(self) -> Array: ...

    @property
    def SWdown(self) -> Array: ...

    @property
    def aqh(self) -> Array: ...

    @property
    def fCori(self) -> Array: ...

    @property
    def wSpeed(self) -> Array: ...


class SurfaceSettings(StaticSettings, Protocol):
    """Surface iteration controls and numerical temperature bounds."""

    @property
    def surfaceTemperatureIterations(self) -> int: ...

    @property
    def minLWdown(self) -> float: ...

    @property
    def minTAir(self) -> float: ...

    @property
    def minTIce(self) -> float: ...

    @property
    def wSpeedMin(self) -> float: ...


class GrowthState(IceThermodynamicState, MaskState, SurfaceState, Protocol):
    """Surface forcing plus transported ice, snow, and ocean coupling fields."""

    @property
    def Qnet(self) -> Array: ...

    @property
    def Qsw(self) -> Array: ...

    @property
    def evap(self) -> Array: ...

    @property
    def ocSalt(self) -> Array: ...

    @property
    def os_hIceMean(self) -> Array: ...

    @property
    def os_hSnowMean(self) -> Array: ...

    @property
    def precip(self) -> Array: ...

    @property
    def runoff(self) -> Array: ...

    @property
    def snowfall(self) -> Array: ...

    @property
    def theta(self) -> Array: ...


class GrowthSettings(SurfaceSettings, MassSettings, StaticSettings, Protocol):
    """Thickness category counts, regularization and timestep controls."""

    @property
    def minActualIceThickness(self) -> float: ...

    @property
    def Area_reg(self) -> float: ...

    @property
    def deltatTherm(self) -> float: ...

    @property
    def hIce_reg(self) -> float: ...

    @property
    def nITC(self) -> int: ...

    @property
    def recip_deltatTherm(self) -> float: ...

    @property
    def recip_nITC(self) -> float: ...
