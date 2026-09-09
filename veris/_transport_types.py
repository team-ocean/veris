"""Read-only inputs for directional sea-ice transport and its face fluxes.

Fields are horizontal JAX arrays, including two halo cells on each boundary.
Settings are static during tracing. Structural properties support both mutable
state containers and immutable PyTrees without imposing runtime inheritance.
"""

from typing import Protocol

from jax import Array

from veris._typing import AreaState, MaskState, StaticSettings, ThicknessState


class FluxSettings(StaticSettings, Protocol):
    """Thermodynamic timestep and bound on the flux-limiter slope ratio."""

    @property
    def use_sharding(self) -> bool: ...

    @property
    def deltatTherm(self) -> float: ...

    @property
    def CrMax(self) -> float: ...


class AdvectionSettings(FluxSettings, StaticSettings, Protocol):
    """Select conservative extensive or intensive transport and halo execution."""

    @property
    def extensiveFld(self) -> bool: ...


class ZonalFluxState(MaskState, Protocol):
    """Velocity, masks and reciprocal spacing on zonal transport faces."""

    @property
    def iceMaskU(self) -> Array: ...

    @property
    def maskInU(self) -> Array: ...

    @property
    def uIce(self) -> Array: ...

    @property
    def recip_dxC(self) -> Array: ...


class MeridionalFluxState(MaskState, Protocol):
    """Velocity, masks and reciprocal spacing on meridional transport faces."""

    @property
    def iceMaskV(self) -> Array: ...

    @property
    def maskInV(self) -> Array: ...

    @property
    def vIce(self) -> Array: ...

    @property
    def recip_dyC(self) -> Array: ...


class TransportState(ZonalFluxState, MeridionalFluxState, Protocol):
    """Face lengths and cell factors for the two directional transport sweeps."""

    @property
    def dyG(self) -> Array: ...

    @property
    def dxG(self) -> Array: ...

    @property
    def maskInC(self) -> Array: ...

    @property
    def recip_rA(self) -> Array: ...

    @property
    def recip_hIceMean(self) -> Array: ...


class AdvectionState(TransportState, AreaState, ThicknessState, Protocol):
    """Transport geometry together with all three advected sea-ice fields."""
