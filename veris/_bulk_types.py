"""Read-only interfaces for standalone atmospheric bulk heat-flux kernels.

The original bulk routines obtain constants from a hashable state.settings
container. Gravity (grav) and Earth radius (radius) are explicit caller inputs;
they are not additional defaults in the sea-ice settings registry. Array inputs
accept both NumPy host arrays and JAX arrays, with dimensions documented by each
kernel. These protocols do not register or change any runtime PyTree type.
"""

from typing import Protocol, TypeVar

from jax import Array

from veris._typing import StaticSettings

HeatFluxes = tuple[Array, Array, Array]
CESMFluxes = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]
LANLFluxes = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]

SettingsT_co = TypeVar("SettingsT_co", covariant=True)


class BulkState(StaticSettings, Protocol[SettingsT_co]):
    """Hashable caller-owned wrapper holding the constants needed by a formula."""

    @property
    def settings(self) -> SettingsT_co: ...


class HeightSettings(Protocol):
    """Virtual-temperature and geopotential constants for atmospheric levels."""

    @property
    def grav(self) -> float: ...

    @property
    def radius(self) -> float: ...

    @property
    def rdair(self) -> float: ...

    @property
    def zvir(self) -> float: ...


class SimpleFluxSettings(StaticSettings, Protocol):
    """Bulk transfer constants for heat flux and its temperature derivative."""

    @property
    def ce(self) -> float: ...

    @property
    def ch(self) -> float: ...

    @property
    def cpdair(self) -> float: ...

    @property
    def latvap(self) -> float: ...

    @property
    def stefBoltz(self) -> float: ...

    @property
    def umin_o(self) -> float: ...


class LongwaveSettings(StaticSettings, Protocol):
    """Ocean longwave emissivity and humidity regularization constants."""

    @property
    def emissivity(self) -> float: ...

    @property
    def eps2(self) -> float: ...

    @property
    def stefBoltz(self) -> float: ...


class CESMFluxSettings(StaticSettings, Protocol):
    """Atmospheric exchange constants and reference heights for CESM fluxes."""

    @property
    def cpdair(self) -> float: ...

    @property
    def cpvir(self) -> float: ...

    @property
    def grav(self) -> float: ...

    @property
    def karman(self) -> float: ...

    @property
    def latvap(self) -> float: ...

    @property
    def stefBoltz(self) -> float: ...

    @property
    def umin_o(self) -> float: ...

    @property
    def zref(self) -> float: ...

    @property
    def ztref(self) -> float: ...

    @property
    def zvir(self) -> float: ...


class LANLFluxSettings(StaticSettings, Protocol):
    """Atmospheric exchange constants for the MITgcm LANL formula."""

    @property
    def cpdair(self) -> float: ...

    @property
    def gamma_blk(self) -> float: ...

    @property
    def grav(self) -> float: ...

    @property
    def karman(self) -> float: ...

    @property
    def latvap(self) -> float: ...

    @property
    def ocean_emissivity(self) -> float: ...

    @property
    def rhoAir(self) -> float: ...

    @property
    def stefBoltz(self) -> float: ...

    @property
    def zvir(self) -> float: ...
