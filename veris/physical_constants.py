"""Registry-backed physical and empirical law constants.

Defaults preserve the legacy settings.py parameterizations, including independently
rounded gas constants and latent heats. Exact dependencies are recomputed on
construction and dataclasses.replace; instances are hashable static JAX arguments.
"""

import math
from dataclasses import dataclass, field
from typing import cast

from veris._metadata import PhysicalConstant, validate_derived, validate_scalars

PHYSICALCONSTANTS: dict[str, PhysicalConstant] = {
    "hCut": PhysicalConstant(
        0.15,
        float,
        "Snow thickness at the transition to optically opaque snow albedo",
        "m",
    ),
    "rhoIce": PhysicalConstant(900.0, float, "density of ice /kg/m3"),
    "rhoFresh": PhysicalConstant(1000.0, float, "density of fresh water /kg/m3"),
    "rhoSea": PhysicalConstant(1026.0, float, "density of sea water /kg/m3"),
    "rhoAir": PhysicalConstant(1.3, float, "density of air /kg/m3"),
    "rhoSnow": PhysicalConstant(330.0, float, "density of snow /kg/m3"),
    "recip_rhoFresh": PhysicalConstant(0.001, float, "1 / rhoFresh /m3/kg"),
    "recip_rhoSea": PhysicalConstant(0.0009746588693957114, float, "1 / rhoSea /m3/kg"),
    "rhoIce2rhoSnow": PhysicalConstant(
        2.727272727272727, float, "Ice-to-snow density ratio (dimensionless)"
    ),
    "rhoIce2rhoFresh": PhysicalConstant(
        0.9, float, "Ice-to-freshwater density ratio (dimensionless)"
    ),
    "rhoFresh2rhoSnow": PhysicalConstant(
        3.0303030303030303, float, "Freshwater-to-snow density ratio (dimensionless)"
    ),
    "dryIceAlb": PhysicalConstant(0.75, float, "albedo of dry ice /-"),
    "dryIceAlb_south": PhysicalConstant(
        0.75, float, "albedo of dry ice in the southern hemisphere /-"
    ),
    "wetIceAlb": PhysicalConstant(0.66, float, "albedo of wet ice /-"),
    "wetIceAlb_south": PhysicalConstant(
        0.66, float, "albedo of wet ice in the southern hemisphere /-"
    ),
    "drySnowAlb": PhysicalConstant(0.84, float, "albedo of dry snow /-"),
    "drySnowAlb_south": PhysicalConstant(
        0.84, float, "albedo of dry snow in the southern hemisphere /-"
    ),
    "wetSnowAlb": PhysicalConstant(0.7, float, "albedo of wet snow /-"),
    "wetSnowAlb_south": PhysicalConstant(
        0.7, float, "albedo of wet snow in the southern hemisphere /-"
    ),
    "wetAlbTemp": PhysicalConstant(
        0.0, float, "temperature above which the wet albedos are used /°C"
    ),
    "lhFusion": PhysicalConstant(334000.0, float, "latent heat of fusion /J/kg"),
    "lhEvap": PhysicalConstant(2500000.0, float, "latent heat of evaporation /J/kg"),
    "lhSublim": PhysicalConstant(2834000.0, float, "latent heat of sublimation /J/kg"),
    "cpAir": PhysicalConstant(1005.0, float, "heat capacity of air /J/kg K"),
    "cpWater": PhysicalConstant(3986.0, float, "heat capacity of water /J/kg K"),
    "stefBoltz": PhysicalConstant(
        5.67e-08, float, "Stefan-Boltzmann constant /W/m^2/K^4"
    ),
    "iceEmiss": PhysicalConstant(0.95, float, "longwave ice emissivity /-"),
    "snowEmiss": PhysicalConstant(0.95, float, "longwave snow emissivity /-"),
    "iceConduct": PhysicalConstant(
        2.1656, float, "Sea ice thermal conductivity /W/m/K"
    ),
    "snowConduct": PhysicalConstant(0.31, float, "Snow thermal conductivity /W/m/K"),
    "shortwave": PhysicalConstant(0.3, float, "shortwave ice penetration factor /-"),
    "tempFrz": PhysicalConstant(-1.96, float, "freezing temperature /°C"),
    "dtempFrz_dS": PhysicalConstant(
        0.0,
        float,
        "Derivative of freezing temperature with respect to salinity /°C/(g/kg)",
    ),
    "saltIce_ref": PhysicalConstant(0.0, float, "reference salinity of sea ice /g/kg"),
    "saltOcn_ref": PhysicalConstant(
        34.7, float, "reference salinity of the ocean /g/kg"
    ),
    "dalton": PhysicalConstant(
        0.00175,
        float,
        "dalton number/ sensible and latent heat transfer coefficient /m/s",
    ),
    "celsius2K": PhysicalConstant(
        273.15, float, "Offset added to Celsius temperature to obtain kelvin /K"
    ),
    "stantonNr": PhysicalConstant(0.0056, float, "stanton number /-"),
    "uStarBase": PhysicalConstant(
        0.0125, float, "typical friction velocity beneath sea ice /m/s"
    ),
    "McPheeTaperFac": PhysicalConstant(
        12.5, float, "tapering factor at the ice bottom /-"
    ),
    "h0": PhysicalConstant(0.5, float, "lead closing parameter"),
    "recip_h0": PhysicalConstant(2.0, float, "1 / h0"),
    "h0_south": PhysicalConstant(
        0.5, float, "lead closing parameter in the southern hemisphere"
    ),
    "recip_h0_south": PhysicalConstant(2.0, float, "1 / h0_south"),
    "airTurnAngle": PhysicalConstant(0.0, float, "turning angle of air-ice stress /°"),
    "waterTurnAngle": PhysicalConstant(
        0.0, float, "turning angle of water-ice stress /°"
    ),
    "sinWat": PhysicalConstant(0.0, float, "sin of waterTurnAngle /-"),
    "cosWat": PhysicalConstant(1.0, float, "cos of waterTurnAngle /-"),
    "airIceDrag": PhysicalConstant(0.0012, float, "air-ice drag coefficient /-"),
    "airIceDrag_south": PhysicalConstant(
        0.0012, float, "air-ice drag coefficient in the southern hemisphere /-"
    ),
    "waterIceDrag": PhysicalConstant(0.0055, float, "water-ice drag coefficient /-"),
    "waterIceDrag_south": PhysicalConstant(
        0.0055, float, "water-ice drag coefficient in the southern hemisphere /-"
    ),
    "gravity": PhysicalConstant(9.81, float, "gravitational acceleration /m/s^2"),
    "PlasDefCoeff": PhysicalConstant(
        2.0, float, "axes ratio of the elliptical yield curve /-"
    ),
    "pStar": PhysicalConstant(27500.0, float, "sea ice strength parameter /Pa"),
    "cStar": PhysicalConstant(20.0, float, "sea ice strength parameter /-"),
    "basalDragU0": PhysicalConstant(5e-05, float, "basal drag parameter /m/s"),
    "basalDragK1": PhysicalConstant(8.0, float, "basal drag parameter /-"),
    "basalDragK2": PhysicalConstant(0.0, float, "basal drag parameter /-"),
    "cBasalStar": PhysicalConstant(20.0, float, "basal drag parameter /-"),
    "tensileStrFac": PhysicalConstant(0.0, float, "sea ice tensile strength factor /-"),
    "sideDragCoeff": PhysicalConstant(0.001, float, "side drag coefficient /-"),
    "sideDragU0": PhysicalConstant(0.01, float, "side drag critical velocity /m/s"),
    "bolzc": PhysicalConstant(1.38065e-23, float, "Boltzmann's constant /J/K/molecule"),
    "avogad": PhysicalConstant(6.02214e26, float, "Avogadro number /molecules/kmole"),
    "rgas": PhysicalConstant(
        8314.47, float, "avogad * bolzc - Ideal gas constant /J/K/kmole"
    ),
    "mwdair": PhysicalConstant(28.966, float, "molecular weight of dry air /kg/kmole"),
    "mwwv": PhysicalConstant(18.016, float, "molecular weight water vapor /kg/kmole"),
    "rdair": PhysicalConstant(
        287.042, float, "RGAS / MWDAIR - dry air gas constant /J/K/kg"
    ),
    "rwv": PhysicalConstant(
        461.505, float, "RGAS / MWWV - water vapor constant /J/K/kg"
    ),
    "zvir": PhysicalConstant(
        0.608,
        float,
        "(RWV / RDAIR) - 1.0 - Dry-air water-vapor molecular mass ratio /-",
    ),
    "cpdair": PhysicalConstant(1004.64, float, "specific heat of dry air /J/K/kg"),
    "cpwv": PhysicalConstant(1810.0, float, "specific heat of water vapor /J/K/kg"),
    "cpvir": PhysicalConstant(0.802, float, "- /-"),
    "karman": PhysicalConstant(0.4, float, "von Karman constant"),
    "latvap": PhysicalConstant(2501000.0, float, "latent heat of evaporation /J/kg"),
    "p0": PhysicalConstant(
        100000.0, float, "reference pressure to compute potential temperature /Pa"
    ),
    "cappa": PhysicalConstant(0.286, float, "R/Cp /-"),
    "zzsice": PhysicalConstant(0.0005, float, "ice surface roughness /m"),
    "ch": PhysicalConstant(
        0.001, float, "bulk transfer coefficient for sensible heat /-"
    ),
    "ce": PhysicalConstant(
        0.00115, float, "bulk transfer coefficient for latent heat /-"
    ),
    "emissivity": PhysicalConstant(1.0, float, "surface emissivity /-"),
    "ocean_emissivity": PhysicalConstant(0.985, float, "ocean surface emissivity /-"),
    "snow_emissivity": PhysicalConstant(0.98, float, "snow surface emissivity /-"),
    "ice_emissivity": PhysicalConstant(0.98, float, "ice surface emissivity /-"),
    "tf0kel": PhysicalConstant(273.15, float, "freezing temp of fresh water /K"),
    "gamma_blk": PhysicalConstant(0.01, float, "adiabatic lapse rate /C/m"),
    "ocean_albedo": PhysicalConstant(0.1, float, "ocean albedo /-"),
    "ice_albedo": PhysicalConstant(0.7, float, "ice albedo /-"),
    "radius": PhysicalConstant(
        6371000.0,
        float,
        "Mean spherical Earth radius used by geopotential-height conversion /m",
    ),
    "iceVaporPressureTemperature": PhysicalConstant(
        2663.5,
        float,
        "Ice saturation vapor-pressure inverse-temperature coefficient",
        "K",
    ),
    "iceVaporPressureLog10Offset": PhysicalConstant(
        12.537, float, "Ice saturation vapor-pressure base-10 logarithmic offset", "1"
    ),
    "waterVaporDryAirMassRatio": PhysicalConstant(
        0.622,
        float,
        "Water-vapor to dry-air molecular mass ratio in saturation laws",
        "1",
    ),
    "iceSurfacePressure": PhysicalConstant(
        100000.0,
        float,
        "Fixed pressure for the ice-surface humidity parameterization",
        "Pa",
    ),
    "iceShortwaveExtinction": PhysicalConstant(
        1.5, float, "Exponential shortwave attenuation coefficient within ice", "m^-1"
    ),
    "McPheeTaperArea": PhysicalConstant(
        0.4, float, "Ice concentration scale of the McPhee bottom-melt taper", "1"
    ),
    "McPheeTaperSteepness": PhysicalConstant(
        7.0,
        float,
        "McPhee bottom-melt taper numerator, divided by McPheeTaperArea",
        "1",
    ),
    "lateralMeltAreaFactor": PhysicalConstant(
        0.5,
        float,
        "Lateral concentration-loss factor multiplying reciprocal ice thickness",
        "1",
    ),
    "cesmSaturationHumidityScale": PhysicalConstant(
        640380.0,
        float,
        "CESM exponential saturation specific-humidity scale",
        "kg m^-3",
    ),
    "cesmSaturationHumidityTemperature": PhysicalConstant(
        5107.4, float, "CESM saturation humidity inverse-temperature coefficient", "K"
    ),
    "augustVaporPressureLog10Offset": PhysicalConstant(
        9.4051, float, "August saturation vapor-pressure logarithmic offset", "1"
    ),
    "augustVaporPressureTemperature": PhysicalConstant(
        2353.0,
        float,
        "August saturation vapor-pressure inverse-temperature coefficient",
        "K",
    ),
    "mmHgToPa": PhysicalConstant(
        133.322,
        float,
        "Conversion from millimetres of mercury to pascals",
        "Pa mmHg^-1",
    ),
    "neutralDragInverseWind": PhysicalConstant(
        0.0027,
        float,
        "Reciprocal-wind coefficient in the neutral ocean drag law",
        "m s^-1",
    ),
    "neutralDragConstant": PhysicalConstant(
        0.000142, float, "Constant coefficient in the neutral ocean drag law", "1"
    ),
    "neutralDragLinearWind": PhysicalConstant(
        7.64e-05,
        float,
        "Linear-wind coefficient in the neutral ocean drag law",
        "s m^-1",
    ),
    "cesmUnstableMomentumOffset": PhysicalConstant(
        1.571, float, "Historically rounded CESM unstable momentum angle offset", "1"
    ),
    "longwaveHumidityPressureScale": PhysicalConstant(
        1000.0,
        float,
        "Humidity pressure scale in the ocean longwave parameterization",
        "hPa",
    ),
    "longwaveClearSkyOffset": PhysicalConstant(
        0.39, float, "Clear-sky offset in the ocean longwave parameterization", "1"
    ),
    "longwaveHumidityCoefficient": PhysicalConstant(
        0.05,
        float,
        "Humidity coefficient in the ocean longwave parameterization",
        "hPa^-0.5",
    ),
    "seawaterHumidityFactor": PhysicalConstant(
        0.98,
        float,
        "Salinity reduction factor for ocean surface saturation humidity",
        "1",
    ),
    "cesmNeutralHeatUnstable": PhysicalConstant(
        0.0327,
        float,
        "CESM unstable neutral heat-transfer square-root coefficient",
        "1",
    ),
    "cesmNeutralHeatStable": PhysicalConstant(
        0.018, float, "CESM stable neutral heat-transfer square-root coefficient", "1"
    ),
    "cesmNeutralMoisture": PhysicalConstant(
        0.0346, float, "CESM neutral moisture-transfer square-root coefficient", "1"
    ),
    "bulkUnstableStabilityCoefficient": PhysicalConstant(
        16.0, float, "Unstable Monin-Obukhov similarity coefficient", "1"
    ),
    "bulkStableStabilityCoefficient": PhysicalConstant(
        5.0, float, "Magnitude of the negative stable similarity coefficient", "1"
    ),
    "lanlSaturationHumidityScale": PhysicalConstant(
        3.797915, float, "LANL saturation specific-humidity scale", "1"
    ),
    "lanlSaturationExponentOffset": PhysicalConstant(
        7.93252e-06,
        float,
        "LANL saturation exponent offset multiplying latent heat",
        "kg J^-1",
    ),
    "lanlSaturationExponentTemperature": PhysicalConstant(
        0.002166847,
        float,
        "LANL inverse-temperature exponent coefficient multiplying latent heat",
        "K kg J^-1",
    ),
    "lanlReferencePressure": PhysicalConstant(
        1013.0,
        float,
        "LANL humidity reference pressure in its original hPa convention",
        "hPa",
    ),
    "longwaveCloudLatitudes": PhysicalConstant(
        (
            -90.0,
            -80.0,
            -70.0,
            -60.0,
            -50.0,
            -40.0,
            -30.0,
            -20.0,
            -10.0,
            -5.0,
            0.0,
            5.0,
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
        ),
        tuple,
        "Latitude knots for the ocean longwave cloud correction",
        "degrees_north",
    ),
    "longwaveCloudCoefficients": PhysicalConstant(
        (
            0.88,
            0.84,
            0.8,
            0.76,
            0.72,
            0.68,
            0.63,
            0.59,
            0.52,
            0.5,
            0.5,
            0.5,
            0.52,
            0.59,
            0.63,
            0.68,
            0.72,
            0.76,
            0.8,
            0.84,
            0.88,
        ),
        tuple,
        "Cloud correction coefficients corresponding to latitude knots",
        "1",
    ),
}


@dataclass(frozen=True)
class PhysicalConstants:
    """Validated immutable physical constants initialized from the registry."""

    rhoIce: float = cast(float, PHYSICALCONSTANTS["rhoIce"].default)
    rhoFresh: float = cast(float, PHYSICALCONSTANTS["rhoFresh"].default)
    rhoSea: float = cast(float, PHYSICALCONSTANTS["rhoSea"].default)
    rhoAir: float = cast(float, PHYSICALCONSTANTS["rhoAir"].default)
    rhoSnow: float = cast(float, PHYSICALCONSTANTS["rhoSnow"].default)
    recip_rhoFresh: float = field(
        default=cast(float, PHYSICALCONSTANTS["recip_rhoFresh"].default), init=False
    )
    recip_rhoSea: float = field(
        default=cast(float, PHYSICALCONSTANTS["recip_rhoSea"].default), init=False
    )
    rhoIce2rhoSnow: float = field(
        default=cast(float, PHYSICALCONSTANTS["rhoIce2rhoSnow"].default), init=False
    )
    rhoIce2rhoFresh: float = field(
        default=cast(float, PHYSICALCONSTANTS["rhoIce2rhoFresh"].default), init=False
    )
    rhoFresh2rhoSnow: float = field(
        default=cast(float, PHYSICALCONSTANTS["rhoFresh2rhoSnow"].default), init=False
    )
    dryIceAlb: float = cast(float, PHYSICALCONSTANTS["dryIceAlb"].default)
    dryIceAlb_south: float = cast(float, PHYSICALCONSTANTS["dryIceAlb_south"].default)
    wetIceAlb: float = cast(float, PHYSICALCONSTANTS["wetIceAlb"].default)
    wetIceAlb_south: float = cast(float, PHYSICALCONSTANTS["wetIceAlb_south"].default)
    drySnowAlb: float = cast(float, PHYSICALCONSTANTS["drySnowAlb"].default)
    drySnowAlb_south: float = cast(float, PHYSICALCONSTANTS["drySnowAlb_south"].default)
    wetSnowAlb: float = cast(float, PHYSICALCONSTANTS["wetSnowAlb"].default)
    wetSnowAlb_south: float = cast(float, PHYSICALCONSTANTS["wetSnowAlb_south"].default)
    wetAlbTemp: float = cast(float, PHYSICALCONSTANTS["wetAlbTemp"].default)
    lhFusion: float = cast(float, PHYSICALCONSTANTS["lhFusion"].default)
    lhEvap: float = cast(float, PHYSICALCONSTANTS["lhEvap"].default)
    lhSublim: float = field(
        default=cast(float, PHYSICALCONSTANTS["lhSublim"].default), init=False
    )
    cpAir: float = cast(float, PHYSICALCONSTANTS["cpAir"].default)
    cpWater: float = cast(float, PHYSICALCONSTANTS["cpWater"].default)
    stefBoltz: float = cast(float, PHYSICALCONSTANTS["stefBoltz"].default)
    iceEmiss: float = cast(float, PHYSICALCONSTANTS["iceEmiss"].default)
    snowEmiss: float = cast(float, PHYSICALCONSTANTS["snowEmiss"].default)
    iceConduct: float = cast(float, PHYSICALCONSTANTS["iceConduct"].default)
    snowConduct: float = cast(float, PHYSICALCONSTANTS["snowConduct"].default)
    shortwave: float = cast(float, PHYSICALCONSTANTS["shortwave"].default)
    tempFrz: float = cast(float, PHYSICALCONSTANTS["tempFrz"].default)
    dtempFrz_dS: float = cast(float, PHYSICALCONSTANTS["dtempFrz_dS"].default)
    saltIce_ref: float = cast(float, PHYSICALCONSTANTS["saltIce_ref"].default)
    saltOcn_ref: float = cast(float, PHYSICALCONSTANTS["saltOcn_ref"].default)
    dalton: float = cast(float, PHYSICALCONSTANTS["dalton"].default)
    celsius2K: float = cast(float, PHYSICALCONSTANTS["celsius2K"].default)
    stantonNr: float = cast(float, PHYSICALCONSTANTS["stantonNr"].default)
    uStarBase: float = cast(float, PHYSICALCONSTANTS["uStarBase"].default)
    McPheeTaperFac: float = cast(float, PHYSICALCONSTANTS["McPheeTaperFac"].default)
    h0: float = cast(float, PHYSICALCONSTANTS["h0"].default)
    recip_h0: float = field(
        default=cast(float, PHYSICALCONSTANTS["recip_h0"].default), init=False
    )
    h0_south: float = cast(float, PHYSICALCONSTANTS["h0_south"].default)
    recip_h0_south: float = field(
        default=cast(float, PHYSICALCONSTANTS["recip_h0_south"].default), init=False
    )
    airTurnAngle: float = cast(float, PHYSICALCONSTANTS["airTurnAngle"].default)
    waterTurnAngle: float = cast(float, PHYSICALCONSTANTS["waterTurnAngle"].default)
    sinWat: float = field(
        default=cast(float, PHYSICALCONSTANTS["sinWat"].default), init=False
    )
    cosWat: float = field(
        default=cast(float, PHYSICALCONSTANTS["cosWat"].default), init=False
    )
    airIceDrag: float = cast(float, PHYSICALCONSTANTS["airIceDrag"].default)
    airIceDrag_south: float = cast(float, PHYSICALCONSTANTS["airIceDrag_south"].default)
    waterIceDrag: float = cast(float, PHYSICALCONSTANTS["waterIceDrag"].default)
    waterIceDrag_south: float = cast(
        float, PHYSICALCONSTANTS["waterIceDrag_south"].default
    )
    gravity: float = cast(float, PHYSICALCONSTANTS["gravity"].default)
    PlasDefCoeff: float = cast(float, PHYSICALCONSTANTS["PlasDefCoeff"].default)
    pStar: float = cast(float, PHYSICALCONSTANTS["pStar"].default)
    cStar: float = cast(float, PHYSICALCONSTANTS["cStar"].default)
    basalDragU0: float = cast(float, PHYSICALCONSTANTS["basalDragU0"].default)
    basalDragK1: float = cast(float, PHYSICALCONSTANTS["basalDragK1"].default)
    basalDragK2: float = cast(float, PHYSICALCONSTANTS["basalDragK2"].default)
    cBasalStar: float = cast(float, PHYSICALCONSTANTS["cBasalStar"].default)
    tensileStrFac: float = cast(float, PHYSICALCONSTANTS["tensileStrFac"].default)
    sideDragCoeff: float = cast(float, PHYSICALCONSTANTS["sideDragCoeff"].default)
    sideDragU0: float = cast(float, PHYSICALCONSTANTS["sideDragU0"].default)
    bolzc: float = cast(float, PHYSICALCONSTANTS["bolzc"].default)
    avogad: float = cast(float, PHYSICALCONSTANTS["avogad"].default)
    rgas: float = cast(float, PHYSICALCONSTANTS["rgas"].default)
    mwdair: float = cast(float, PHYSICALCONSTANTS["mwdair"].default)
    mwwv: float = cast(float, PHYSICALCONSTANTS["mwwv"].default)
    rdair: float = cast(float, PHYSICALCONSTANTS["rdair"].default)
    rwv: float = cast(float, PHYSICALCONSTANTS["rwv"].default)
    zvir: float = cast(float, PHYSICALCONSTANTS["zvir"].default)
    cpdair: float = cast(float, PHYSICALCONSTANTS["cpdair"].default)
    cpwv: float = cast(float, PHYSICALCONSTANTS["cpwv"].default)
    cpvir: float = cast(float, PHYSICALCONSTANTS["cpvir"].default)
    karman: float = cast(float, PHYSICALCONSTANTS["karman"].default)
    latvap: float = cast(float, PHYSICALCONSTANTS["latvap"].default)
    p0: float = cast(float, PHYSICALCONSTANTS["p0"].default)
    cappa: float = cast(float, PHYSICALCONSTANTS["cappa"].default)
    zzsice: float = cast(float, PHYSICALCONSTANTS["zzsice"].default)
    ch: float = cast(float, PHYSICALCONSTANTS["ch"].default)
    ce: float = cast(float, PHYSICALCONSTANTS["ce"].default)
    emissivity: float = cast(float, PHYSICALCONSTANTS["emissivity"].default)
    ocean_emissivity: float = cast(float, PHYSICALCONSTANTS["ocean_emissivity"].default)
    snow_emissivity: float = cast(float, PHYSICALCONSTANTS["snow_emissivity"].default)
    ice_emissivity: float = cast(float, PHYSICALCONSTANTS["ice_emissivity"].default)
    tf0kel: float = cast(float, PHYSICALCONSTANTS["tf0kel"].default)
    gamma_blk: float = cast(float, PHYSICALCONSTANTS["gamma_blk"].default)
    ocean_albedo: float = cast(float, PHYSICALCONSTANTS["ocean_albedo"].default)
    ice_albedo: float = cast(float, PHYSICALCONSTANTS["ice_albedo"].default)
    radius: float = cast(float, PHYSICALCONSTANTS["radius"].default)

    iceVaporPressureTemperature: float = cast(
        float, PHYSICALCONSTANTS["iceVaporPressureTemperature"].default
    )
    iceVaporPressureLog10Offset: float = cast(
        float, PHYSICALCONSTANTS["iceVaporPressureLog10Offset"].default
    )
    waterVaporDryAirMassRatio: float = cast(
        float, PHYSICALCONSTANTS["waterVaporDryAirMassRatio"].default
    )
    iceSurfacePressure: float = cast(
        float, PHYSICALCONSTANTS["iceSurfacePressure"].default
    )
    iceShortwaveExtinction: float = cast(
        float, PHYSICALCONSTANTS["iceShortwaveExtinction"].default
    )
    McPheeTaperArea: float = cast(float, PHYSICALCONSTANTS["McPheeTaperArea"].default)
    McPheeTaperSteepness: float = cast(
        float, PHYSICALCONSTANTS["McPheeTaperSteepness"].default
    )
    lateralMeltAreaFactor: float = cast(
        float, PHYSICALCONSTANTS["lateralMeltAreaFactor"].default
    )
    cesmSaturationHumidityScale: float = cast(
        float, PHYSICALCONSTANTS["cesmSaturationHumidityScale"].default
    )
    cesmSaturationHumidityTemperature: float = cast(
        float, PHYSICALCONSTANTS["cesmSaturationHumidityTemperature"].default
    )
    augustVaporPressureLog10Offset: float = cast(
        float, PHYSICALCONSTANTS["augustVaporPressureLog10Offset"].default
    )
    augustVaporPressureTemperature: float = cast(
        float, PHYSICALCONSTANTS["augustVaporPressureTemperature"].default
    )
    mmHgToPa: float = cast(float, PHYSICALCONSTANTS["mmHgToPa"].default)
    neutralDragInverseWind: float = cast(
        float, PHYSICALCONSTANTS["neutralDragInverseWind"].default
    )
    neutralDragConstant: float = cast(
        float, PHYSICALCONSTANTS["neutralDragConstant"].default
    )
    neutralDragLinearWind: float = cast(
        float, PHYSICALCONSTANTS["neutralDragLinearWind"].default
    )
    cesmUnstableMomentumOffset: float = cast(
        float, PHYSICALCONSTANTS["cesmUnstableMomentumOffset"].default
    )
    longwaveHumidityPressureScale: float = cast(
        float, PHYSICALCONSTANTS["longwaveHumidityPressureScale"].default
    )
    longwaveClearSkyOffset: float = cast(
        float, PHYSICALCONSTANTS["longwaveClearSkyOffset"].default
    )
    longwaveHumidityCoefficient: float = cast(
        float, PHYSICALCONSTANTS["longwaveHumidityCoefficient"].default
    )
    seawaterHumidityFactor: float = cast(
        float, PHYSICALCONSTANTS["seawaterHumidityFactor"].default
    )
    cesmNeutralHeatUnstable: float = cast(
        float, PHYSICALCONSTANTS["cesmNeutralHeatUnstable"].default
    )
    cesmNeutralHeatStable: float = cast(
        float, PHYSICALCONSTANTS["cesmNeutralHeatStable"].default
    )
    cesmNeutralMoisture: float = cast(
        float, PHYSICALCONSTANTS["cesmNeutralMoisture"].default
    )
    bulkUnstableStabilityCoefficient: float = cast(
        float, PHYSICALCONSTANTS["bulkUnstableStabilityCoefficient"].default
    )
    bulkStableStabilityCoefficient: float = cast(
        float, PHYSICALCONSTANTS["bulkStableStabilityCoefficient"].default
    )
    lanlSaturationHumidityScale: float = cast(
        float, PHYSICALCONSTANTS["lanlSaturationHumidityScale"].default
    )
    lanlSaturationExponentOffset: float = cast(
        float, PHYSICALCONSTANTS["lanlSaturationExponentOffset"].default
    )
    lanlSaturationExponentTemperature: float = cast(
        float, PHYSICALCONSTANTS["lanlSaturationExponentTemperature"].default
    )
    lanlReferencePressure: float = cast(
        float, PHYSICALCONSTANTS["lanlReferencePressure"].default
    )
    longwaveCloudLatitudes: tuple[float, ...] = cast(
        tuple[float, ...], PHYSICALCONSTANTS["longwaveCloudLatitudes"].default
    )
    longwaveCloudCoefficients: tuple[float, ...] = cast(
        tuple[float, ...], PHYSICALCONSTANTS["longwaveCloudCoefficients"].default
    )

    hCut: float = cast(float, PHYSICALCONSTANTS["hCut"].default)

    def __post_init__(self) -> None:
        """Validate host scalars and recompute exact dependent quantities."""
        validate_scalars(
            self,
            PHYSICALCONSTANTS,
            positive=frozenset(
                [
                    "hCut",
                    "rhoIce",
                    "rhoFresh",
                    "rhoSea",
                    "rhoAir",
                    "rhoSnow",
                    "lhFusion",
                    "lhEvap",
                    "cpAir",
                    "cpWater",
                    "stefBoltz",
                    "iceConduct",
                    "snowConduct",
                    "h0",
                    "h0_south",
                    "gravity",
                    "PlasDefCoeff",
                    "basalDragU0",
                    "basalDragK1",
                    "sideDragU0",
                    "bolzc",
                    "avogad",
                    "rgas",
                    "mwdair",
                    "mwwv",
                    "rdair",
                    "rwv",
                    "zvir",
                    "cpdair",
                    "cpwv",
                    "karman",
                    "latvap",
                    "p0",
                    "zzsice",
                    "radius",
                    "iceSurfacePressure",
                    "McPheeTaperArea",
                    "lanlReferencePressure",
                ]
            ),
        )
        object.__setattr__(self, "recip_rhoFresh", 1.0 / self.rhoFresh)
        object.__setattr__(self, "recip_rhoSea", 1.0 / self.rhoSea)
        object.__setattr__(self, "rhoIce2rhoSnow", self.rhoIce / self.rhoSnow)
        object.__setattr__(self, "rhoIce2rhoFresh", self.rhoIce / self.rhoFresh)
        object.__setattr__(self, "rhoFresh2rhoSnow", self.rhoFresh / self.rhoSnow)
        object.__setattr__(self, "lhSublim", self.lhFusion + self.lhEvap)
        object.__setattr__(self, "recip_h0", 1.0 / self.h0)
        object.__setattr__(self, "recip_h0_south", 1.0 / self.h0_south)
        object.__setattr__(self, "sinWat", math.sin(math.radians(self.waterTurnAngle)))
        object.__setattr__(self, "cosWat", math.cos(math.radians(self.waterTurnAngle)))
        if self.tensileStrFac <= -1:
            raise ValueError("tensileStrFac must exceed -1")
        validate_derived(
            self,
            (
                "recip_rhoFresh",
                "recip_rhoSea",
                "rhoIce2rhoSnow",
                "rhoIce2rhoFresh",
                "rhoFresh2rhoSnow",
                "lhSublim",
                "recip_h0",
                "recip_h0_south",
                "sinWat",
                "cosWat",
            ),
        )
        if len(self.longwaveCloudLatitudes) < 2:
            raise ValueError("longwaveCloudLatitudes must contain at least two knots")
        if len(self.longwaveCloudLatitudes) != len(self.longwaveCloudCoefficients):
            raise ValueError(
                "longwaveCloudCoefficients must match longwaveCloudLatitudes length"
            )
        if any(
            right <= left
            for left, right in zip(
                self.longwaveCloudLatitudes[:-1],
                self.longwaveCloudLatitudes[1:],
                strict=True,
            )
        ):
            raise ValueError("longwaveCloudLatitudes must be strictly increasing")
