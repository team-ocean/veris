"""Registry-backed execution and numerical model configuration.

Defaults reproduce legacy settings.py. Host validation prevents invalid static
arguments from reaching compiled kernels. Reciprocal values follow immutable
updates through dataclasses.replace.
"""

from dataclasses import dataclass, field
from typing import cast

from veris._metadata import Setting, validate_derived, validate_scalars

SETTINGS: dict[str, Setting] = {
    "nx": Setting(8, int, "Local interior grid extent along the x direction", "1"),
    "ny": Setting(12, int, "Local interior grid extent along the y direction", "1"),
    "artificialGridSpacing": Setting(
        8000.0, float, "Uniform Cartesian grid spacing in the artificial example", "m"
    ),
    "artificialWindSpeed": Setting(
        5.0, float, "Prescribed signed zonal wind in the artificial example", "m s-1"
    ),
    "artificialAirTemperature": Setting(
        260.0,
        float,
        "Prescribed atmosphere and initial ice-surface temperature in the artificial example",
        "K",
    ),
    "artificialIceThickness": Setting(
        1.0,
        float,
        "Initial grid-cell mean ice thickness over ocean in the artificial example",
        "m",
    ),
    "artificialSnowThickness": Setting(
        0.05,
        float,
        "Initial grid-cell mean snow thickness over ocean in the artificial example",
        "m",
    ),
    "artificialIceArea": Setting(
        0.8,
        float,
        "Initial ocean-cell ice concentration in the artificial example",
        "1",
    ),
    "artificialOceanDepth": Setting(
        -100.0, float, "Signed ocean bottom elevation in the artificial example", "m"
    ),
    "artificialCoriolis": Setting(
        0.0001, float, "Uniform Coriolis frequency in the artificial example", "s-1"
    ),
    "artificialCooling": Setting(
        100.0,
        float,
        "Default upward open-water cooling imposed each artificial step",
        "W m-2",
    ),
    "artificialTimeStep": Setting(
        600.0,
        float,
        "Default dynamics and thermodynamics timestep for the artificial example",
        "s",
    ),
    "artificialEVPsteps": Setting(
        5, int, "Default EVP substeps in the artificial example", "1"
    ),
    "geometrySurfaceTemperature": Setting(
        273.0, float, "Initial surface temperature used by the geometry adapter /K"
    ),
    "printEvpResidual": Setting(
        False, bool, "Print EVP residual diagnostics during execution"
    ),
    "deltatTherm": Setting(86400.0, float, "timestep for thermodynamic equations /s"),
    "recip_deltatTherm": Setting(
        1.1574074074074073e-05, float, "Reciprocal thermodynamic timestep /s^-1"
    ),
    "deltatDyn": Setting(86400.0, float, "timestep for dynamic equations /s"),
    "recip_deltatDyn": Setting(
        1.1574074074074073e-05, float, "Reciprocal dynamic timestep /s^-1"
    ),
    "nITC": Setting(5, int, "number of ice thickness categories /-"),
    "recip_nITC": Setting(0.2, float, "1 / nITC /-"),
    "noSlip": Setting(True, bool, "flag for using the no-slip condition"),
    "useRelativeWind": Setting(
        True,
        bool,
        "Use wind minus ice velocity for stress; otherwise use wind velocity",
    ),
    "secondOrderBC": Setting(
        False,
        bool,
        "flag for using the second order approximation for boundary conditions",
    ),
    "extensiveFld": Setting(
        True, bool, "flag whether the advective fields are extensive"
    ),
    "useRealFreshWaterFlux": Setting(
        False,
        bool,
        "flag for using the sea ice load in the calculation of the ocean surface height",
    ),
    "useFreedrift": Setting(False, bool, "flag for using the freedrift solver"),
    "useEVP": Setting(True, bool, "flag for using the EVP solver"),
    "evpAlpha": Setting(500.0, float, "EVP parameter /-"),
    "evpBeta": Setting(500.0, float, "EVP parameter /-"),
    "useAdaptiveEVP": Setting(
        False, bool, "flag for using adaptive relaxation parameters"
    ),
    "aEVPalphaMin": Setting(5.0, float, "lower limit of alpha and beta /-"),
    "aEvpCoeff": Setting(
        0.5, float, "largest stabilized frequency for adaptive EVP /-"
    ),
    "explicitDrag": Setting(
        True,
        bool,
        "flag for stepping the momentum equation in a explicit or implicit way",
    ),
    "nEVPsteps": Setting(
        400, int, "number of sub-cycling iterations of the EVP solver"
    ),
    "computeEvpResidual": Setting(
        False,
        bool,
        "flag for computing the residual of stress and velocity in the EVP loop",
    ),
    "use_coastline": Setting(
        False, bool, "flag for using the coastline data for lateral drag"
    ),
    "use_sharding": Setting(
        True, bool, "flag for using parallel execution via sharded arrays"
    ),
    "minLWdown": Setting(60.0, float, "minimum downward longwave radiation /W/m^2"),
    "maxTIce": Setting(30.0, float, "maximum ice temperature /°C"),
    "minTIce": Setting(-50.0, float, "minimum ice temperature /°C"),
    "minTAir": Setting(-50.0, float, "minimum air temperature /°C"),
    "Area_reg": Setting(
        0.0225, float, "Squared ice-concentration regularization (dimensionless)"
    ),
    "hIce_reg": Setting(
        0.010000000000000002, float, "regularization value for the ice thickness /m^2"
    ),
    "wSpeedMin": Setting(1e-10, float, "minimum wind speed /m/s"),
    "hIce_min": Setting(1e-05, float, "'minimum' ice thickness /m"),
    "Area_min": Setting(1e-05, float, "'minimum' ice cover fraction /-"),
    "cDragMin": Setting(0.25, float, "minimum of linear ice-ocean drag coefficient /-"),
    "seaIceLoadFac": Setting(1.0, float, "factor to scale sea ice loading /-"),
    "deltaMin": Setting(2e-09, float, "minimum value of delta /-"),
    "pressReplFac": Setting(1.0, float, "flag whether to use replacement pressure /-"),
    "CrMax": Setting(1000000.0, float, "advective flux parameter /-"),
    "umin_o": Setting(0.5, float, "minimum atm. wind speed over ocean surface /m/s"),
    "umin_i": Setting(1.0, float, "minimum atm. wind speed over ice surface /m/s"),
    "zref": Setting(10.0, float, "reference height for wind speed /m"),
    "ztref": Setting(2.0, float, "reference height for air temperature /m"),
    "eps2": Setting(1e-20, float, "threshold value /-"),
    "surfaceTemperatureIterations": Setting(
        6, int, "Number of Newton iterations in the ice surface energy balance", "1"
    ),
    "minActualIceThickness": Setting(
        0.05, float, "Minimum actual ice thickness used in thermodynamic growth", "m"
    ),
    "basalDragSmoothing": Setting(
        10.0,
        float,
        "Inverse thickness scale of the basal-drag smooth positive part",
        "m^-1",
    ),
    "basalDragMinArea": Setting(
        0.01, float, "Minimum ice concentration that enables basal drag", "1"
    ),
    "aEVPmassMin": Setting(
        0.0001,
        float,
        "Minimum cell ice mass used in adaptive EVP relaxation",
        "kg m^-2",
    ),
    "aEVPcStar": Setting(4.0, float, "Adaptive EVP relaxation multiplier", "1"),
    "evpStressRelaxation": Setting(
        1.0, float, "Independent normal stress damping coefficient in EVP updates", "1"
    ),
    "evpShearRelaxation": Setting(
        0.25, float, "Independent shear stress forcing coefficient in EVP updates", "1"
    ),
    "bulkStabilityLimit": Setting(
        10.0, float, "Maximum absolute height-to-Obukhov-length ratio", "1"
    ),
    "lanlMinWindSpeed": Setting(
        1.0, float, "Minimum open-ocean wind speed in LANL bulk fluxes", "m s^-1"
    ),
    "lanlBulkIterations": Setting(
        5, int, "Number of LANL Monin-Obukhov stability iterations", "1"
    ),
}


@dataclass(frozen=True)
class Settings:
    """Validated immutable model settings initialized from the registry."""

    deltatTherm: float = SETTINGS["deltatTherm"].default
    recip_deltatTherm: float = field(
        default=SETTINGS["recip_deltatTherm"].default, init=False
    )
    deltatDyn: float = SETTINGS["deltatDyn"].default
    recip_deltatDyn: float = field(
        default=SETTINGS["recip_deltatDyn"].default, init=False
    )
    nITC: int = cast(int, SETTINGS["nITC"].default)
    recip_nITC: float = field(default=SETTINGS["recip_nITC"].default, init=False)
    noSlip: bool = cast(bool, SETTINGS["noSlip"].default)
    useRelativeWind: bool = cast(bool, SETTINGS["useRelativeWind"].default)
    secondOrderBC: bool = cast(bool, SETTINGS["secondOrderBC"].default)
    extensiveFld: bool = cast(bool, SETTINGS["extensiveFld"].default)
    useRealFreshWaterFlux: bool = cast(bool, SETTINGS["useRealFreshWaterFlux"].default)
    useFreedrift: bool = cast(bool, SETTINGS["useFreedrift"].default)
    useEVP: bool = cast(bool, SETTINGS["useEVP"].default)
    evpAlpha: float = SETTINGS["evpAlpha"].default
    evpBeta: float = SETTINGS["evpBeta"].default
    useAdaptiveEVP: bool = cast(bool, SETTINGS["useAdaptiveEVP"].default)
    aEVPalphaMin: float = SETTINGS["aEVPalphaMin"].default
    aEvpCoeff: float = SETTINGS["aEvpCoeff"].default
    explicitDrag: bool = cast(bool, SETTINGS["explicitDrag"].default)
    nEVPsteps: int = cast(int, SETTINGS["nEVPsteps"].default)
    computeEvpResidual: bool = cast(bool, SETTINGS["computeEvpResidual"].default)
    printEvpResidual: bool = cast(bool, SETTINGS["printEvpResidual"].default)
    geometrySurfaceTemperature: float = SETTINGS["geometrySurfaceTemperature"].default
    use_coastline: bool = cast(bool, SETTINGS["use_coastline"].default)
    use_sharding: bool = cast(bool, SETTINGS["use_sharding"].default)
    minLWdown: float = SETTINGS["minLWdown"].default
    maxTIce: float = SETTINGS["maxTIce"].default
    minTIce: float = SETTINGS["minTIce"].default
    minTAir: float = SETTINGS["minTAir"].default
    Area_reg: float = SETTINGS["Area_reg"].default
    hIce_reg: float = SETTINGS["hIce_reg"].default
    wSpeedMin: float = SETTINGS["wSpeedMin"].default
    hIce_min: float = SETTINGS["hIce_min"].default
    Area_min: float = SETTINGS["Area_min"].default
    cDragMin: float = SETTINGS["cDragMin"].default
    seaIceLoadFac: float = SETTINGS["seaIceLoadFac"].default
    deltaMin: float = SETTINGS["deltaMin"].default
    pressReplFac: float = SETTINGS["pressReplFac"].default
    CrMax: float = SETTINGS["CrMax"].default
    umin_o: float = SETTINGS["umin_o"].default
    umin_i: float = SETTINGS["umin_i"].default
    zref: float = SETTINGS["zref"].default
    ztref: float = SETTINGS["ztref"].default
    eps2: float = SETTINGS["eps2"].default

    surfaceTemperatureIterations: int = cast(
        int, SETTINGS["surfaceTemperatureIterations"].default
    )
    minActualIceThickness: float = SETTINGS["minActualIceThickness"].default
    basalDragSmoothing: float = SETTINGS["basalDragSmoothing"].default
    basalDragMinArea: float = SETTINGS["basalDragMinArea"].default
    aEVPmassMin: float = SETTINGS["aEVPmassMin"].default
    aEVPcStar: float = SETTINGS["aEVPcStar"].default
    evpStressRelaxation: float = SETTINGS["evpStressRelaxation"].default
    evpShearRelaxation: float = SETTINGS["evpShearRelaxation"].default
    bulkStabilityLimit: float = SETTINGS["bulkStabilityLimit"].default
    lanlMinWindSpeed: float = SETTINGS["lanlMinWindSpeed"].default
    lanlBulkIterations: int = cast(int, SETTINGS["lanlBulkIterations"].default)

    nx: int = cast(int, SETTINGS["nx"].default)
    ny: int = cast(int, SETTINGS["ny"].default)
    artificialGridSpacing: float = SETTINGS["artificialGridSpacing"].default
    artificialWindSpeed: float = SETTINGS["artificialWindSpeed"].default
    artificialAirTemperature: float = SETTINGS["artificialAirTemperature"].default
    artificialIceThickness: float = SETTINGS["artificialIceThickness"].default
    artificialSnowThickness: float = SETTINGS["artificialSnowThickness"].default
    artificialIceArea: float = SETTINGS["artificialIceArea"].default
    artificialOceanDepth: float = SETTINGS["artificialOceanDepth"].default
    artificialCoriolis: float = SETTINGS["artificialCoriolis"].default
    artificialCooling: float = SETTINGS["artificialCooling"].default
    artificialTimeStep: float = SETTINGS["artificialTimeStep"].default
    artificialEVPsteps: int = cast(int, SETTINGS["artificialEVPsteps"].default)

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
                    "hIce_reg",
                    "wSpeedMin",
                    "hIce_min",
                    "Area_min",
                    "deltaMin",
                    "CrMax",
                    "umin_o",
                    "umin_i",
                    "zref",
                    "ztref",
                    "surfaceTemperatureIterations",
                    "minActualIceThickness",
                    "basalDragSmoothing",
                    "basalDragMinArea",
                    "aEVPmassMin",
                    "aEVPcStar",
                    "evpStressRelaxation",
                    "evpShearRelaxation",
                    "bulkStabilityLimit",
                    "lanlMinWindSpeed",
                    "lanlBulkIterations",
                    "eps2",
                    "artificialGridSpacing",
                    "artificialAirTemperature",
                    "artificialTimeStep",
                    "artificialEVPsteps",
                ]
            ),
        )
        if self.Area_reg < 0:
            raise ValueError("Area_reg must be nonnegative")
        object.__setattr__(self, "recip_deltatTherm", 1.0 / self.deltatTherm)
        object.__setattr__(self, "recip_deltatDyn", 1.0 / self.deltatDyn)
        object.__setattr__(self, "recip_nITC", 1.0 / self.nITC)
        if self.minTIce > self.maxTIce:
            raise ValueError("minTIce must not exceed maxTIce")
        validate_derived(self, ("recip_deltatTherm", "recip_deltatDyn", "recip_nITC"))
        for name in ("nx", "ny"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least two interior cells")
        for name in ("artificialIceThickness", "artificialSnowThickness"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.artificialIceArea <= 1:
            raise ValueError("artificialIceArea must lie between zero and one")
        if self.artificialOceanDepth > 0:
            raise ValueError("artificialOceanDepth must be nonpositive")
