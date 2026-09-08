"""Read-only structural inputs for sea-ice momentum and constitutive kernels.

Horizontal JAX fields include halo cells where required by a stencil. Each
protocol lists only fields consumed by its kernel and transitive callees;
composition supports mutable state containers and immutable JAX PyTrees.
Physical settings are static scalar coefficients during JIT tracing.
"""

from typing import Protocol

from jax import Array

from veris._typing import AreaState, BoundarySettings, MaskState, StaticSettings


class IceThicknessState(Protocol):
    """Grid-cell mean ice thickness in meters."""

    @property
    def hIceMean(self) -> Array: ...


class FaceMaskState(Protocol):
    """Ocean masks at velocity faces."""

    @property
    def iceMaskU(self) -> Array: ...

    @property
    def iceMaskV(self) -> Array: ...


class InteriorFaceMaskState(Protocol):
    """Interior masks on velocity faces."""

    @property
    def maskInU(self) -> Array: ...

    @property
    def maskInV(self) -> Array: ...


class OceanVelocityState(Protocol):
    """Ocean velocity and Coriolis parameter."""

    @property
    def uOcean(self) -> Array: ...

    @property
    def vOcean(self) -> Array: ...

    @property
    def fCori(self) -> Array: ...


class StrengthState(IceThicknessState, AreaState, MaskState, Protocol):
    """Fields determining the maximum compressive ice stress."""


class StrengthSettings(StaticSettings, Protocol):
    """Empirical pressure scale and concentration sensitivity."""

    @property
    def pStar(self) -> float: ...

    @property
    def cStar(self) -> float: ...


class WaterDragSettings(StaticSettings, Protocol):
    """Hemispheric water-ice drag coefficients and seawater density."""

    @property
    def waterIceDrag_south(self) -> float: ...

    @property
    def waterIceDrag(self) -> float: ...

    @property
    def rhoSea(self) -> float: ...


class OceanDragState(OceanVelocityState, InteriorFaceMaskState, MaskState, Protocol):
    """Ocean velocities and masks used in linearized water drag."""


class OceanDragSettings(WaterDragSettings, StaticSettings, Protocol):
    """Water drag parameters including the minimum linear coefficient."""

    @property
    def cDragMin(self) -> float: ...


class BasalDragState(IceThicknessState, AreaState, InteriorFaceMaskState, Protocol):
    """Ice thickness, concentration and bathymetry for keel drag."""

    @property
    def R_low(self) -> Array: ...


class BasalDragSettings(StaticSettings, Protocol):
    """Keel geometry, speed regularization and concentration parameters."""

    @property
    def basalDragK2(self) -> float: ...

    @property
    def basalDragU0(self) -> float: ...

    @property
    def basalDragK1(self) -> float: ...

    @property
    def cBasalStar(self) -> float: ...


class SideDragState(FaceMaskState, Protocol):
    """Face concentration, ice mass and prescribed coastline factors."""

    @property
    def AreaW(self) -> Array: ...

    @property
    def AreaS(self) -> Array: ...

    @property
    def Fu(self) -> Array: ...

    @property
    def Fv(self) -> Array: ...

    @property
    def SeaIceMassU(self) -> Array: ...

    @property
    def SeaIceMassV(self) -> Array: ...


class SideDragSettings(StaticSettings, Protocol):
    """Lateral drag strength, regularization and coastline selection."""

    @property
    def use_coastline(self) -> bool: ...

    @property
    def sideDragCoeff(self) -> float: ...

    @property
    def sideDragU0(self) -> float: ...


class StrainState(MaskState, FaceMaskState, Protocol):
    """Metric factors and masks for center and corner strain rates."""

    @property
    def recip_dxU(self) -> Array: ...

    @property
    def recip_dyV(self) -> Array: ...

    @property
    def k2AtC(self) -> Array: ...

    @property
    def maskInC(self) -> Array: ...

    @property
    def k1AtC(self) -> Array: ...

    @property
    def recip_dyU(self) -> Array: ...

    @property
    def recip_dxV(self) -> Array: ...

    @property
    def k1AtZ(self) -> Array: ...

    @property
    def k2AtZ(self) -> Array: ...


class StrainSettings(BoundarySettings, StaticSettings, Protocol):
    """Lateral slip condition and order of boundary correction."""

    @property
    def secondOrderBC(self) -> bool: ...


class ViscosityState(Protocol):
    """Cell geometry and compressive strength for the ice rheology."""

    @property
    def rAz(self) -> Array: ...

    @property
    def recip_rA(self) -> Array: ...

    @property
    def SeaIceStrength(self) -> Array: ...


class ViscositySettings(StaticSettings, Protocol):
    """Elliptical yield curve, regularization and pressure parameters."""

    @property
    def PlasDefCoeff(self) -> float: ...

    @property
    def deltaMin(self) -> float: ...

    @property
    def tensileStrFac(self) -> float: ...

    @property
    def pressReplFac(self) -> float: ...


class StressDivergenceState(Protocol):
    """Face lengths and reciprocal areas for the stress divergence."""

    @property
    def dyV(self) -> Array: ...

    @property
    def dxV(self) -> Array: ...

    @property
    def recip_rAu(self) -> Array: ...

    @property
    def dxU(self) -> Array: ...

    @property
    def dyU(self) -> Array: ...

    @property
    def recip_rAv(self) -> Array: ...


class FreeDriftState(IceThicknessState, OceanVelocityState, FaceMaskState, Protocol):
    """Forcing and geometry for the stress-free momentum balance."""

    @property
    def WindForcingX(self) -> Array: ...

    @property
    def WindForcingY(self) -> Array: ...


class FreeDriftSettings(WaterDragSettings, StaticSettings, Protocol):
    """Material parameters for the stress-free momentum balance."""

    @property
    def rhoIce(self) -> float: ...


class OceanStressState(OceanDragState, Protocol):
    """Ice and ocean velocities for the surface stress."""

    @property
    def uIce(self) -> Array: ...

    @property
    def vIce(self) -> Array: ...


class OceanStressSettings(OceanDragSettings, StaticSettings, Protocol):
    """Water drag parameters and turning angle in degrees."""

    @property
    def waterTurnAngle(self) -> float: ...
