"""Registry-backed execution and numerical model configuration.

Defaults reproduce legacy settings.py. Host validation prevents invalid static
arguments from reaching compiled kernels. Reciprocal values follow immutable
updates through dataclasses.replace.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from veris._metadata import (
    FROM_REGISTRY,
    registry_defaults,
    validate_derived,
    validate_scalars,
)


class Setting(NamedTuple):
    """Default, scalar type and human-readable description of a model setting."""

    default: float | int | bool | str
    type: type[float] | type[int] | type[bool] | type[str]
    description: str
    units: str = ""


SETTINGS: dict[str, Setting] = {
    "dtype": Setting(
        "float64", str, "Model floating-point precision: float32 or float64", "-"
    ),
    "nx": Setting(8, int, "Local interior grid extent along the x direction", "1"),
    "ny": Setting(12, int, "Local interior grid extent along the y direction", "1"),
    "geometrySurfaceTemperature": Setting(
        273.0,
        float,
        "Initial surface temperature used by the geometry adapter",
        ":math:`K`",
    ),
    "printEvpResidual": Setting(
        False, bool, "Print EVP residual diagnostics during execution", "-"
    ),
    "deltatTherm": Setting(
        86400.0, float, "timestep for thermodynamic equations", ":math:`s`"
    ),
    "recip_deltatTherm": Setting(
        1.1574074074074073e-05,
        float,
        "Reciprocal thermodynamic timestep",
        ":math:`s^{-1}`",
    ),
    "deltatDyn": Setting(86400.0, float, "timestep for dynamic equations", ":math:`s`"),
    "recip_deltatDyn": Setting(
        1.1574074074074073e-05, float, "Reciprocal dynamic timestep", ":math:`s^{-1}`"
    ),
    "nITC": Setting(5, int, "number of ice thickness categories", "-"),
    "recip_nITC": Setting(0.2, float, "1 / nITC", "-"),
    "noSlip": Setting(True, bool, "flag for using the no-slip condition", "-"),
    "useRelativeWind": Setting(
        True,
        bool,
        "Use wind minus ice velocity for stress; otherwise use wind velocity",
        "-",
    ),
    "secondOrderBC": Setting(
        False,
        bool,
        "flag for using the second order approximation for boundary conditions",
        "-",
    ),
    "extensiveFld": Setting(
        True,
        bool,
        "flag whether the advective fields are extensive",
        "-",
    ),
    "useRealFreshWaterFlux": Setting(
        False,
        bool,
        "flag for using the sea ice load in the calculation of the ocean surface height",
        "-",
    ),
    "useFreedrift": Setting(False, bool, "flag for using the freedrift solver", "-"),
    "useEVP": Setting(True, bool, "flag for using the EVP solver", "-"),
    "evpAlpha": Setting(500.0, float, "EVP parameter", "-"),
    "evpBeta": Setting(500.0, float, "EVP parameter", "-"),
    "useAdaptiveEVP": Setting(
        False, bool, "flag for using adaptive relaxation parameters", "-"
    ),
    "aEVPalphaMin": Setting(5.0, float, "lower limit of alpha and beta", "-"),
    "aEvpCoeff": Setting(
        0.5, float, "largest stabilized frequency for adaptive EVP", "-"
    ),
    "explicitDrag": Setting(
        True,
        bool,
        "Reserved legacy explicit-drag flag; currently unused by the solvers",
        "-",
    ),
    "nEVPsteps": Setting(
        400, int, "number of sub-cycling iterations of the EVP solver", "-"
    ),
    "computeEvpResidual": Setting(
        False,
        bool,
        "flag for computing the residual of stress and velocity in the EVP loop",
        "-",
    ),
    "use_coastline": Setting(
        False, bool, "flag for using the coastline data for lateral drag", "-"
    ),
    "use_sharding": Setting(
        True, bool, "flag for using parallel execution via sharded arrays", "-"
    ),
    "CrMax": Setting(
        1000000.0, float, "Absolute cap on the advected-field slope ratio", "-"
    ),
    "eps2": Setting(
        1e-20,
        float,
        "Additive safeguard for the longwave humidity-pressure square root",
        ":math:`hPa`",
    ),
    "surfaceTemperatureIterations": Setting(
        6, int, "Number of Newton iterations in the ice surface energy balance", "1"
    ),
    "aEVPmassMin": Setting(
        0.0001,
        float,
        "Minimum cell ice mass used in adaptive EVP relaxation",
        ":math:`kg\\,m^{-2}`",
    ),
    "aEVPcStar": Setting(4.0, float, "Adaptive EVP relaxation multiplier", "1"),
    "lanlBulkIterations": Setting(
        5, int, "Number of LANL Monin-Obukhov stability iterations", "1"
    ),
}


__all__ = ["SETTINGS", "Configuration", "Setting"]


@dataclass(frozen=True)
@registry_defaults(SETTINGS)
class Configuration:
    """Validated immutable model settings initialized from the registry."""

    dtype: str = field(default=FROM_REGISTRY, kw_only=True)

    deltatTherm: float = FROM_REGISTRY
    recip_deltatTherm: float = field(init=False)
    deltatDyn: float = FROM_REGISTRY
    recip_deltatDyn: float = field(init=False)
    nITC: int = FROM_REGISTRY
    recip_nITC: float = field(init=False)
    noSlip: bool = FROM_REGISTRY
    useRelativeWind: bool = FROM_REGISTRY
    secondOrderBC: bool = FROM_REGISTRY
    extensiveFld: bool = FROM_REGISTRY
    useRealFreshWaterFlux: bool = FROM_REGISTRY
    useFreedrift: bool = FROM_REGISTRY
    useEVP: bool = FROM_REGISTRY
    evpAlpha: float = FROM_REGISTRY
    evpBeta: float = FROM_REGISTRY
    useAdaptiveEVP: bool = FROM_REGISTRY
    aEVPalphaMin: float = FROM_REGISTRY
    aEvpCoeff: float = FROM_REGISTRY
    explicitDrag: bool = FROM_REGISTRY
    nEVPsteps: int = FROM_REGISTRY
    computeEvpResidual: bool = FROM_REGISTRY
    printEvpResidual: bool = FROM_REGISTRY
    geometrySurfaceTemperature: float = FROM_REGISTRY
    use_coastline: bool = FROM_REGISTRY
    use_sharding: bool = FROM_REGISTRY
    CrMax: float = FROM_REGISTRY
    eps2: float = FROM_REGISTRY

    surfaceTemperatureIterations: int = FROM_REGISTRY
    aEVPmassMin: float = FROM_REGISTRY
    aEVPcStar: float = FROM_REGISTRY
    lanlBulkIterations: int = FROM_REGISTRY

    nx: int = FROM_REGISTRY
    ny: int = FROM_REGISTRY

    def __post_init__(self) -> None:
        """Validate host scalars and recompute exact dependent quantities."""
        validate_scalars(
            self,
            SETTINGS,
            positive=frozenset(
                [
                    "deltatTherm",
                    "deltatDyn",
                    "nITC",
                    "nEVPsteps",
                    "evpAlpha",
                    "evpBeta",
                    "aEVPalphaMin",
                    "aEvpCoeff",
                    "CrMax",
                    "surfaceTemperatureIterations",
                    "aEVPmassMin",
                    "aEVPcStar",
                    "lanlBulkIterations",
                    "eps2",
                ]
            ),
        )
        object.__setattr__(self, "recip_deltatTherm", 1.0 / float(self.deltatTherm))
        object.__setattr__(self, "recip_deltatDyn", 1.0 / float(self.deltatDyn))
        object.__setattr__(self, "recip_nITC", 1.0 / float(self.nITC))
        validate_derived(self, ("recip_deltatTherm", "recip_deltatDyn", "recip_nITC"))
        for name in ("nx", "ny"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least two interior cells")
