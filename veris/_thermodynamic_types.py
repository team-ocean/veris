"""Structural fields and constants for ice/snow thermodynamics.

Surface flux inputs are horizontal JAX arrays; Growth also consumes transported
ice/snow means and prescribed freshwater/ocean heat forcing. GrowthSettings
includes solve4temp's constants because each thickness category calls that
solver. nITC is an integer static loop/shape count. Properties preserve custom
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
    """Albedos, conductivities, exchange coefficients, and temperature limits."""

    @property
    def celsius2K(self) -> float: ...

    @property
    def cpAir(self) -> float: ...

    @property
    def dalton(self) -> float: ...

    @property
    def dryIceAlb(self) -> float: ...

    @property
    def dryIceAlb_south(self) -> float: ...

    @property
    def drySnowAlb(self) -> float: ...

    @property
    def drySnowAlb_south(self) -> float: ...

    @property
    def hCut(self) -> float: ...

    @property
    def iceConduct(self) -> float: ...

    @property
    def iceEmiss(self) -> float: ...

    @property
    def lhSublim(self) -> float: ...

    @property
    def minLWdown(self) -> float: ...

    @property
    def minTAir(self) -> float: ...

    @property
    def minTIce(self) -> float: ...

    @property
    def rhoAir(self) -> float: ...

    @property
    def shortwave(self) -> float: ...

    @property
    def snowConduct(self) -> float: ...

    @property
    def snowEmiss(self) -> float: ...

    @property
    def stefBoltz(self) -> float: ...

    @property
    def wSpeedMin(self) -> float: ...

    @property
    def wetAlbTemp(self) -> float: ...

    @property
    def wetIceAlb(self) -> float: ...

    @property
    def wetIceAlb_south(self) -> float: ...

    @property
    def wetSnowAlb(self) -> float: ...

    @property
    def wetSnowAlb_south(self) -> float: ...


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
    """Surface-solver constants plus growth, precipitation, and salt conversion."""

    @property
    def Area_reg(self) -> float: ...

    @property
    def McPheeTaperFac(self) -> float: ...

    @property
    def cpWater(self) -> float: ...

    @property
    def deltatTherm(self) -> float: ...

    @property
    def dtempFrz_dS(self) -> float: ...

    @property
    def hIce_reg(self) -> float: ...

    @property
    def lhFusion(self) -> float: ...

    @property
    def nITC(self) -> int: ...

    @property
    def recip_deltatTherm(self) -> float: ...

    @property
    def recip_h0(self) -> float: ...

    @property
    def recip_h0_south(self) -> float: ...

    @property
    def recip_nITC(self) -> float: ...

    @property
    def recip_rhoSea(self) -> float: ...

    @property
    def rhoFresh(self) -> float: ...

    @property
    def rhoFresh2rhoSnow(self) -> float: ...

    @property
    def rhoIce2rhoFresh(self) -> float: ...

    @property
    def rhoIce2rhoSnow(self) -> float: ...

    @property
    def rhoSea(self) -> float: ...

    @property
    def saltIce_ref(self) -> float: ...

    @property
    def stantonNr(self) -> float: ...

    @property
    def tempFrz(self) -> float: ...

    @property
    def uStarBase(self) -> float: ...
