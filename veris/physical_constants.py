"""Registry-backed physical and empirical law constants.

Defaults preserve the legacy settings.py parameterizations, including independently
rounded gas constants and latent heats. Exact dependencies are recomputed on
construction and dataclasses.replace; instances are hashable static JAX arguments.
"""

import math
from dataclasses import dataclass, field

from veris._metadata import (
    FROM_REGISTRY,
    registry_defaults,
    validate_derived,
    validate_scalars,
)
from veris._typing import Parameter
from veris.configuration import SETTINGS

PHYSICALCONSTANTS: dict[str, Parameter] = {
    "pressReplFac": Parameter(
        1.0,
        float,
        "Weight of strain-dependent replacement pressure in the ice constitutive law",
        "1",
    ),
    "evpStressRelaxation": Parameter(
        1.0,
        float,
        "Stress damping coefficient setting the EVP constitutive equilibrium",
        "1",
    ),
    "evpShearRelaxation": Parameter(
        0.25,
        float,
        "Deviatoric stress forcing coefficient setting the EVP constitutive equilibrium",
        "1",
    ),
    "minLWdown": Parameter(
        60.0, float, "minimum downward longwave radiation", ":math:`W/m^2`"
    ),
    "maxTIce": Parameter(
        30.0, float, "maximum ice temperature", ":math:`{}^\\circ\\,C`"
    ),
    "minTIce": Parameter(
        -50.0, float, "minimum ice temperature", ":math:`{}^\\circ\\,C`"
    ),
    "minTAir": Parameter(
        -50.0, float, "minimum air temperature", ":math:`{}^\\circ\\,C`"
    ),
    "Area_reg": Parameter(
        0.0225,
        float,
        "Squared ice-concentration regularization (dimensionless)",
        "-",
    ),
    "hIce_reg": Parameter(
        0.010000000000000002,
        float,
        "regularization value for the ice thickness",
        ":math:`m^2`",
    ),
    "wSpeedMin": Parameter(1e-10, float, "minimum wind speed", ":math:`m/s`"),
    "hIce_min": Parameter(1e-05, float, "'minimum' ice thickness", ":math:`m`"),
    "Area_min": Parameter(1e-05, float, "'minimum' ice cover fraction", "-"),
    "cDragMin": Parameter(
        0.25,
        float,
        "minimum of linear ice-ocean drag coefficient",
        ":math:`kg/(m^2\\,s)`",
    ),
    "seaIceLoadFac": Parameter(1.0, float, "factor to scale sea ice loading", "-"),
    "deltaMin": Parameter(
        2e-09, float, "Minimum strain-rate invariant", ":math:`s^{-1}`"
    ),
    "umin_o": Parameter(
        0.5, float, "minimum atm. wind speed over ocean surface", ":math:`m/s`"
    ),
    "umin_i": Parameter(
        1.0, float, "minimum atm. wind speed over ice surface", ":math:`m/s`"
    ),
    "zref": Parameter(10.0, float, "reference height for wind speed", ":math:`m`"),
    "ztref": Parameter(2.0, float, "reference height for air temperature", ":math:`m`"),
    "minActualIceThickness": Parameter(
        0.05,
        float,
        "Minimum actual ice thickness used in thermodynamic growth",
        ":math:`m`",
    ),
    "basalDragSmoothing": Parameter(
        10.0,
        float,
        "Inverse thickness scale of the basal-drag smooth positive part",
        ":math:`m^{-1}`",
    ),
    "basalDragMinArea": Parameter(
        0.01, float, "Minimum ice concentration that enables basal drag", "1"
    ),
    "bulkStabilityLimit": Parameter(
        10.0, float, "Maximum absolute height-to-Obukhov-length ratio", "1"
    ),
    "lanlMinWindSpeed": Parameter(
        1.0,
        float,
        "Minimum open-ocean wind speed in LANL bulk fluxes",
        ":math:`m\\,s^{-1}`",
    ),
    "hCut": Parameter(
        0.15,
        float,
        "Snow thickness at the transition to optically opaque snow albedo",
        ":math:`m`",
    ),
    "rhoIce": Parameter(900.0, float, "density of ice", ":math:`kg/m^3`"),
    "rhoFresh": Parameter(1000.0, float, "density of fresh water", ":math:`kg/m^3`"),
    "rhoSea": Parameter(1026.0, float, "density of sea water", ":math:`kg/m^3`"),
    "rhoAir": Parameter(1.3, float, "density of air", ":math:`kg/m^3`"),
    "rhoSnow": Parameter(330.0, float, "density of snow", ":math:`kg/m^3`"),
    "recip_rhoFresh": Parameter(0.001, float, "1 / rhoFresh", ":math:`m^3/kg`"),
    "recip_rhoSea": Parameter(
        0.0009746588693957114, float, "1 / rhoSea", ":math:`m^3/kg`"
    ),
    "rhoIce2rhoSnow": Parameter(
        2.727272727272727, float, "Ice-to-snow density ratio (dimensionless)", "-"
    ),
    "rhoIce2rhoFresh": Parameter(
        0.9, float, "Ice-to-freshwater density ratio (dimensionless)", "-"
    ),
    "rhoFresh2rhoSnow": Parameter(
        3.0303030303030303,
        float,
        "Freshwater-to-snow density ratio (dimensionless)",
        "-",
    ),
    "dryIceAlb": Parameter(0.75, float, "albedo of dry ice", "-"),
    "dryIceAlb_south": Parameter(
        0.75, float, "albedo of dry ice in the southern hemisphere", "-"
    ),
    "wetIceAlb": Parameter(0.66, float, "albedo of wet ice", "-"),
    "wetIceAlb_south": Parameter(
        0.66, float, "albedo of wet ice in the southern hemisphere", "-"
    ),
    "drySnowAlb": Parameter(0.84, float, "albedo of dry snow", "-"),
    "drySnowAlb_south": Parameter(
        0.84, float, "albedo of dry snow in the southern hemisphere", "-"
    ),
    "wetSnowAlb": Parameter(0.7, float, "albedo of wet snow", "-"),
    "wetSnowAlb_south": Parameter(
        0.7, float, "albedo of wet snow in the southern hemisphere", "-"
    ),
    "wetAlbTemp": Parameter(
        0.0,
        float,
        "temperature above which the wet albedos are used",
        ":math:`{}^\\circ\\,C`",
    ),
    "lhFusion": Parameter(334000.0, float, "latent heat of fusion", ":math:`J/kg`"),
    "lhEvap": Parameter(2500000.0, float, "latent heat of evaporation", ":math:`J/kg`"),
    "lhSublim": Parameter(
        2834000.0, float, "latent heat of sublimation", ":math:`J/kg`"
    ),
    "cpAir": Parameter(1005.0, float, "heat capacity of air", ":math:`J/(kg\\,K)`"),
    "cpWater": Parameter(3986.0, float, "heat capacity of water", ":math:`J/(kg\\,K)`"),
    "stefBoltz": Parameter(
        5.67e-08, float, "Stefan-Boltzmann constant", ":math:`W/m^2/K^4`"
    ),
    "iceEmiss": Parameter(0.95, float, "longwave ice emissivity", "-"),
    "snowEmiss": Parameter(0.95, float, "longwave snow emissivity", "-"),
    "iceConduct": Parameter(
        2.1656, float, "Sea ice thermal conductivity", ":math:`W/m/K`"
    ),
    "snowConduct": Parameter(0.31, float, "Snow thermal conductivity", ":math:`W/m/K`"),
    "shortwave": Parameter(0.3, float, "shortwave ice penetration factor", "-"),
    "tempFrz": Parameter(-1.96, float, "freezing temperature", ":math:`{}^\\circ\\,C`"),
    "dtempFrz_dS": Parameter(
        0.0,
        float,
        "Derivative of freezing temperature with respect to salinity",
        ":math:`{}^\\circ\\,C/(g/kg)`",
    ),
    "saltIce_ref": Parameter(
        0.0, float, "reference salinity of sea ice", ":math:`g/kg`"
    ),
    "dalton": Parameter(
        0.00175,
        float,
        "Dalton number for sensible and latent heat transfer",
        "-",
    ),
    "celsius2K": Parameter(
        273.15,
        float,
        "Offset added to Celsius temperature to obtain kelvin",
        ":math:`K`",
    ),
    "stantonNr": Parameter(0.0056, float, "stanton number", "-"),
    "uStarBase": Parameter(
        0.0125, float, "typical friction velocity beneath sea ice", ":math:`m/s`"
    ),
    "McPheeTaperFac": Parameter(12.5, float, "tapering factor at the ice bottom", "-"),
    "h0": Parameter(0.5, float, "Lead-closing ice thickness", ":math:`m`"),
    "recip_h0": Parameter(2.0, float, "1 / h0", ":math:`m^{-1}`"),
    "h0_south": Parameter(
        0.5, float, "Lead-closing ice thickness in the southern hemisphere", ":math:`m`"
    ),
    "recip_h0_south": Parameter(2.0, float, "1 / h0_south", ":math:`m^{-1}`"),
    "airTurnAngle": Parameter(
        0.0, float, "turning angle of air-ice stress", ":math:`{}^\\circ`"
    ),
    "waterTurnAngle": Parameter(
        0.0, float, "turning angle of water-ice stress", ":math:`{}^\\circ`"
    ),
    "sinWat": Parameter(0.0, float, "sin of waterTurnAngle", "-"),
    "cosWat": Parameter(1.0, float, "cos of waterTurnAngle", "-"),
    "airIceDrag": Parameter(0.0012, float, "air-ice drag coefficient", "-"),
    "airIceDrag_south": Parameter(
        0.0012, float, "air-ice drag coefficient in the southern hemisphere", "-"
    ),
    "waterIceDrag": Parameter(0.0055, float, "water-ice drag coefficient", "-"),
    "waterIceDrag_south": Parameter(
        0.0055, float, "water-ice drag coefficient in the southern hemisphere", "-"
    ),
    "gravity": Parameter(9.81, float, "gravitational acceleration", ":math:`m/s^2`"),
    "PlasDefCoeff": Parameter(
        2.0, float, "axes ratio of the elliptical yield curve", "-"
    ),
    "pStar": Parameter(27500.0, float, "sea ice strength parameter", ":math:`Pa`"),
    "cStar": Parameter(20.0, float, "sea ice strength parameter", "-"),
    "basalDragU0": Parameter(5e-05, float, "basal drag parameter", ":math:`m/s`"),
    "basalDragK1": Parameter(8.0, float, "basal drag parameter", "-"),
    "basalDragK2": Parameter(
        0.0, float, "Basal stress per unit keel excess thickness", ":math:`Pa/m`"
    ),
    "cBasalStar": Parameter(20.0, float, "basal drag parameter", "-"),
    "tensileStrFac": Parameter(0.0, float, "sea ice tensile strength factor", "-"),
    "sideDragCoeff": Parameter(
        0.001, float, "Coastal drag acceleration coefficient", ":math:`m/s^2`"
    ),
    "sideDragU0": Parameter(0.01, float, "side drag critical velocity", ":math:`m/s`"),
    "bolzc": Parameter(
        1.38065e-23, float, "Boltzmann's constant", ":math:`J/K/molecule`"
    ),
    "avogad": Parameter(
        6.02214e26, float, "Avogadro number", ":math:`molecules/kmole`"
    ),
    "rgas": Parameter(
        8314.47, float, "avogad * bolzc - Ideal gas constant", ":math:`J/K/kmole`"
    ),
    "mwdair": Parameter(
        28.966, float, "molecular weight of dry air", ":math:`kg/kmole`"
    ),
    "mwwv": Parameter(
        18.016, float, "molecular weight water vapor", ":math:`kg/kmole`"
    ),
    "rdair": Parameter(
        287.042, float, "RGAS / MWDAIR - dry air gas constant", ":math:`J/K/kg`"
    ),
    "rwv": Parameter(
        461.505, float, "RGAS / MWWV - water vapor constant", ":math:`J/K/kg`"
    ),
    "zvir": Parameter(
        0.608,
        float,
        "(RWV / RDAIR) - 1.0 - Dry-air water-vapor molecular mass ratio",
        "-",
    ),
    "cpdair": Parameter(1004.64, float, "specific heat of dry air", ":math:`J/K/kg`"),
    "cpwv": Parameter(1810.0, float, "specific heat of water vapor", ":math:`J/K/kg`"),
    "cpvir": Parameter(
        0.802,
        float,
        "Humidity correction to dry-air specific heat (cpwv / cpdair - 1)",
        "-",
    ),
    "karman": Parameter(0.4, float, "von Karman constant", "-"),
    "latvap": Parameter(2501000.0, float, "latent heat of evaporation", ":math:`J/kg`"),
    "p0": Parameter(
        100000.0,
        float,
        "reference pressure to compute potential temperature",
        ":math:`Pa`",
    ),
    "cappa": Parameter(0.286, float, "R/Cp", "-"),
    "zzsice": Parameter(0.0005, float, "ice surface roughness", ":math:`m`"),
    "ch": Parameter(0.001, float, "bulk transfer coefficient for sensible heat", "-"),
    "ce": Parameter(0.00115, float, "bulk transfer coefficient for latent heat", "-"),
    "emissivity": Parameter(1.0, float, "surface emissivity", "-"),
    "ocean_emissivity": Parameter(0.985, float, "ocean surface emissivity", "-"),
    "snow_emissivity": Parameter(0.98, float, "snow surface emissivity", "-"),
    "ice_emissivity": Parameter(0.98, float, "ice surface emissivity", "-"),
    "tf0kel": Parameter(273.15, float, "freezing temp of fresh water", ":math:`K`"),
    "gamma_blk": Parameter(
        0.01, float, "adiabatic lapse rate", ":math:`{}^\\circ\\,C/m`"
    ),
    "ocean_albedo": Parameter(0.1, float, "ocean albedo", "-"),
    "ice_albedo": Parameter(0.7, float, "ice albedo", "-"),
    "radius": Parameter(
        6371000.0,
        float,
        "Mean spherical Earth radius used by geopotential-height conversion",
        ":math:`m`",
    ),
    "iceVaporPressureTemperature": Parameter(
        2663.5,
        float,
        "Ice saturation vapor-pressure inverse-temperature coefficient",
        ":math:`K`",
    ),
    "iceVaporPressureLog10Offset": Parameter(
        12.537, float, "Ice saturation vapor-pressure base-10 logarithmic offset", "1"
    ),
    "waterVaporDryAirMassRatio": Parameter(
        0.622,
        float,
        "Water-vapor to dry-air molecular mass ratio in saturation laws",
        "1",
    ),
    "iceSurfacePressure": Parameter(
        100000.0,
        float,
        "Fixed pressure for the ice-surface humidity parameterization",
        ":math:`Pa`",
    ),
    "iceShortwaveExtinction": Parameter(
        1.5,
        float,
        "Exponential shortwave attenuation coefficient within ice",
        ":math:`m^{-1}`",
    ),
    "McPheeTaperArea": Parameter(
        0.4, float, "Ice concentration scale of the McPhee bottom-melt taper", "1"
    ),
    "McPheeTaperSteepness": Parameter(
        7.0,
        float,
        "McPhee bottom-melt taper numerator, divided by McPheeTaperArea",
        "1",
    ),
    "lateralMeltAreaFactor": Parameter(
        0.5,
        float,
        "Lateral concentration-loss factor multiplying reciprocal ice thickness",
        "1",
    ),
    "cesmSaturationHumidityScale": Parameter(
        640380.0,
        float,
        "CESM exponential saturation specific-humidity scale",
        ":math:`kg\\,m^{-3}`",
    ),
    "cesmSaturationHumidityTemperature": Parameter(
        5107.4,
        float,
        "CESM saturation humidity inverse-temperature coefficient",
        ":math:`K`",
    ),
    "augustVaporPressureLog10Offset": Parameter(
        9.4051, float, "August saturation vapor-pressure logarithmic offset", "1"
    ),
    "augustVaporPressureTemperature": Parameter(
        2353.0,
        float,
        "August saturation vapor-pressure inverse-temperature coefficient",
        ":math:`K`",
    ),
    "mmHgToPa": Parameter(
        133.322,
        float,
        "Conversion from millimetres of mercury to pascals",
        ":math:`Pa\\,mmHg^{-1}`",
    ),
    "neutralDragInverseWind": Parameter(
        0.0027,
        float,
        "Reciprocal-wind coefficient in the neutral ocean drag law",
        ":math:`m\\,s^{-1}`",
    ),
    "neutralDragConstant": Parameter(
        0.000142, float, "Constant coefficient in the neutral ocean drag law", "1"
    ),
    "neutralDragLinearWind": Parameter(
        7.64e-05,
        float,
        "Linear-wind coefficient in the neutral ocean drag law",
        ":math:`s\\,m^{-1}`",
    ),
    "cesmUnstableMomentumOffset": Parameter(
        1.571, float, "Historically rounded CESM unstable momentum angle offset", "1"
    ),
    "longwaveHumidityPressureScale": Parameter(
        1000.0,
        float,
        "Humidity pressure scale in the ocean longwave parameterization",
        ":math:`hPa`",
    ),
    "longwaveClearSkyOffset": Parameter(
        0.39, float, "Clear-sky offset in the ocean longwave parameterization", "1"
    ),
    "longwaveHumidityCoefficient": Parameter(
        0.05,
        float,
        "Humidity coefficient in the ocean longwave parameterization",
        ":math:`hPa^{-0.5}`",
    ),
    "seawaterHumidityFactor": Parameter(
        0.98,
        float,
        "Salinity reduction factor for ocean surface saturation humidity",
        "1",
    ),
    "cesmNeutralHeatUnstable": Parameter(
        0.0327,
        float,
        "CESM unstable neutral heat-transfer square-root coefficient",
        "1",
    ),
    "cesmNeutralHeatStable": Parameter(
        0.018, float, "CESM stable neutral heat-transfer square-root coefficient", "1"
    ),
    "cesmNeutralMoisture": Parameter(
        0.0346, float, "CESM neutral moisture-transfer square-root coefficient", "1"
    ),
    "bulkUnstableStabilityCoefficient": Parameter(
        16.0, float, "Unstable Monin-Obukhov similarity coefficient", "1"
    ),
    "bulkStableStabilityCoefficient": Parameter(
        5.0, float, "Magnitude of the negative stable similarity coefficient", "1"
    ),
    "lanlSaturationHumidityScale": Parameter(
        3.797915, float, "LANL saturation specific-humidity scale", "1"
    ),
    "lanlSaturationExponentOffset": Parameter(
        7.93252e-06,
        float,
        "LANL saturation exponent offset multiplying latent heat",
        ":math:`kg\\,J^{-1}`",
    ),
    "lanlSaturationExponentTemperature": Parameter(
        0.002166847,
        float,
        "LANL inverse-temperature exponent coefficient multiplying latent heat",
        ":math:`K\\,kg\\,J^{-1}`",
    ),
    "lanlReferencePressure": Parameter(
        1013.0,
        float,
        "LANL humidity reference pressure in its original hPa convention",
        ":math:`hPa`",
    ),
    "longwaveCloudLatitudes": Parameter(
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
        ":math:`{}^\\circ\\mathrm{N}`",
    ),
    "longwaveCloudCoefficients": Parameter(
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


__all__ = ["PHYSICALCONSTANTS", "Parameter", "PhysicalConstants"]


@dataclass(frozen=True)
@registry_defaults({"dtype": SETTINGS["dtype"], **PHYSICALCONSTANTS})
class PhysicalConstants:
    """Validated immutable physical constants initialized from the registry."""

    dtype: str = field(default=FROM_REGISTRY, kw_only=True)

    pressReplFac: float = FROM_REGISTRY
    evpStressRelaxation: float = FROM_REGISTRY
    evpShearRelaxation: float = FROM_REGISTRY
    Area_min: float = FROM_REGISTRY
    Area_reg: float = FROM_REGISTRY
    basalDragMinArea: float = FROM_REGISTRY
    basalDragSmoothing: float = FROM_REGISTRY
    bulkStabilityLimit: float = FROM_REGISTRY
    cDragMin: float = FROM_REGISTRY
    deltaMin: float = FROM_REGISTRY
    hIce_min: float = FROM_REGISTRY
    hIce_reg: float = FROM_REGISTRY
    lanlMinWindSpeed: float = FROM_REGISTRY
    maxTIce: float = FROM_REGISTRY
    minActualIceThickness: float = FROM_REGISTRY
    minLWdown: float = FROM_REGISTRY
    minTAir: float = FROM_REGISTRY
    minTIce: float = FROM_REGISTRY
    seaIceLoadFac: float = FROM_REGISTRY
    umin_i: float = FROM_REGISTRY
    umin_o: float = FROM_REGISTRY
    wSpeedMin: float = FROM_REGISTRY
    zref: float = FROM_REGISTRY
    ztref: float = FROM_REGISTRY
    rhoIce: float = FROM_REGISTRY
    rhoFresh: float = FROM_REGISTRY
    rhoSea: float = FROM_REGISTRY
    rhoAir: float = FROM_REGISTRY
    rhoSnow: float = FROM_REGISTRY
    recip_rhoFresh: float = field(init=False)
    recip_rhoSea: float = field(init=False)
    rhoIce2rhoSnow: float = field(init=False)
    rhoIce2rhoFresh: float = field(init=False)
    rhoFresh2rhoSnow: float = field(init=False)
    dryIceAlb: float = FROM_REGISTRY
    dryIceAlb_south: float = FROM_REGISTRY
    wetIceAlb: float = FROM_REGISTRY
    wetIceAlb_south: float = FROM_REGISTRY
    drySnowAlb: float = FROM_REGISTRY
    drySnowAlb_south: float = FROM_REGISTRY
    wetSnowAlb: float = FROM_REGISTRY
    wetSnowAlb_south: float = FROM_REGISTRY
    wetAlbTemp: float = FROM_REGISTRY
    lhFusion: float = FROM_REGISTRY
    lhEvap: float = FROM_REGISTRY
    lhSublim: float = field(init=False)
    cpAir: float = FROM_REGISTRY
    cpWater: float = FROM_REGISTRY
    stefBoltz: float = FROM_REGISTRY
    iceEmiss: float = FROM_REGISTRY
    snowEmiss: float = FROM_REGISTRY
    iceConduct: float = FROM_REGISTRY
    snowConduct: float = FROM_REGISTRY
    shortwave: float = FROM_REGISTRY
    tempFrz: float = FROM_REGISTRY
    dtempFrz_dS: float = FROM_REGISTRY
    saltIce_ref: float = FROM_REGISTRY
    dalton: float = FROM_REGISTRY
    celsius2K: float = FROM_REGISTRY
    stantonNr: float = FROM_REGISTRY
    uStarBase: float = FROM_REGISTRY
    McPheeTaperFac: float = FROM_REGISTRY
    h0: float = FROM_REGISTRY
    recip_h0: float = field(init=False)
    h0_south: float = FROM_REGISTRY
    recip_h0_south: float = field(init=False)
    airTurnAngle: float = FROM_REGISTRY
    waterTurnAngle: float = FROM_REGISTRY
    sinWat: float = field(init=False)
    cosWat: float = field(init=False)
    airIceDrag: float = FROM_REGISTRY
    airIceDrag_south: float = FROM_REGISTRY
    waterIceDrag: float = FROM_REGISTRY
    waterIceDrag_south: float = FROM_REGISTRY
    gravity: float = FROM_REGISTRY
    PlasDefCoeff: float = FROM_REGISTRY
    pStar: float = FROM_REGISTRY
    cStar: float = FROM_REGISTRY
    basalDragU0: float = FROM_REGISTRY
    basalDragK1: float = FROM_REGISTRY
    basalDragK2: float = FROM_REGISTRY
    cBasalStar: float = FROM_REGISTRY
    tensileStrFac: float = FROM_REGISTRY
    sideDragCoeff: float = FROM_REGISTRY
    sideDragU0: float = FROM_REGISTRY
    bolzc: float = FROM_REGISTRY
    avogad: float = FROM_REGISTRY
    rgas: float = FROM_REGISTRY
    mwdair: float = FROM_REGISTRY
    mwwv: float = FROM_REGISTRY
    rdair: float = FROM_REGISTRY
    rwv: float = FROM_REGISTRY
    zvir: float = FROM_REGISTRY
    cpdair: float = FROM_REGISTRY
    cpwv: float = FROM_REGISTRY
    cpvir: float = FROM_REGISTRY
    karman: float = FROM_REGISTRY
    latvap: float = FROM_REGISTRY
    p0: float = FROM_REGISTRY
    cappa: float = FROM_REGISTRY
    zzsice: float = FROM_REGISTRY
    ch: float = FROM_REGISTRY
    ce: float = FROM_REGISTRY
    emissivity: float = FROM_REGISTRY
    ocean_emissivity: float = FROM_REGISTRY
    snow_emissivity: float = FROM_REGISTRY
    ice_emissivity: float = FROM_REGISTRY
    tf0kel: float = FROM_REGISTRY
    gamma_blk: float = FROM_REGISTRY
    ocean_albedo: float = FROM_REGISTRY
    ice_albedo: float = FROM_REGISTRY
    radius: float = FROM_REGISTRY

    iceVaporPressureTemperature: float = FROM_REGISTRY
    iceVaporPressureLog10Offset: float = FROM_REGISTRY
    waterVaporDryAirMassRatio: float = FROM_REGISTRY
    iceSurfacePressure: float = FROM_REGISTRY
    iceShortwaveExtinction: float = FROM_REGISTRY
    McPheeTaperArea: float = FROM_REGISTRY
    McPheeTaperSteepness: float = FROM_REGISTRY
    lateralMeltAreaFactor: float = FROM_REGISTRY
    cesmSaturationHumidityScale: float = FROM_REGISTRY
    cesmSaturationHumidityTemperature: float = FROM_REGISTRY
    augustVaporPressureLog10Offset: float = FROM_REGISTRY
    augustVaporPressureTemperature: float = FROM_REGISTRY
    mmHgToPa: float = FROM_REGISTRY
    neutralDragInverseWind: float = FROM_REGISTRY
    neutralDragConstant: float = FROM_REGISTRY
    neutralDragLinearWind: float = FROM_REGISTRY
    cesmUnstableMomentumOffset: float = FROM_REGISTRY
    longwaveHumidityPressureScale: float = FROM_REGISTRY
    longwaveClearSkyOffset: float = FROM_REGISTRY
    longwaveHumidityCoefficient: float = FROM_REGISTRY
    seawaterHumidityFactor: float = FROM_REGISTRY
    cesmNeutralHeatUnstable: float = FROM_REGISTRY
    cesmNeutralHeatStable: float = FROM_REGISTRY
    cesmNeutralMoisture: float = FROM_REGISTRY
    bulkUnstableStabilityCoefficient: float = FROM_REGISTRY
    bulkStableStabilityCoefficient: float = FROM_REGISTRY
    lanlSaturationHumidityScale: float = FROM_REGISTRY
    lanlSaturationExponentOffset: float = FROM_REGISTRY
    lanlSaturationExponentTemperature: float = FROM_REGISTRY
    lanlReferencePressure: float = FROM_REGISTRY
    longwaveCloudLatitudes: tuple[float, ...] = FROM_REGISTRY
    longwaveCloudCoefficients: tuple[float, ...] = FROM_REGISTRY

    hCut: float = FROM_REGISTRY

    def __post_init__(self) -> None:
        """Validate host scalars and recompute exact dependent quantities."""
        validate_scalars(
            self,
            PHYSICALCONSTANTS,
            positive=frozenset(
                [
                    "evpStressRelaxation",
                    "evpShearRelaxation",
                    "Area_min",
                    "basalDragMinArea",
                    "basalDragSmoothing",
                    "bulkStabilityLimit",
                    "deltaMin",
                    "hIce_min",
                    "hIce_reg",
                    "lanlMinWindSpeed",
                    "minActualIceThickness",
                    "umin_i",
                    "umin_o",
                    "wSpeedMin",
                    "zref",
                    "ztref",
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
        if self.Area_reg < 0:
            raise ValueError("Area_reg must be nonnegative")
        if self.minTIce > self.maxTIce:
            raise ValueError("minTIce must not exceed maxTIce")
        object.__setattr__(self, "recip_rhoFresh", 1.0 / float(self.rhoFresh))
        object.__setattr__(self, "recip_rhoSea", 1.0 / float(self.rhoSea))
        object.__setattr__(
            self, "rhoIce2rhoSnow", float(self.rhoIce) / float(self.rhoSnow)
        )
        object.__setattr__(
            self, "rhoIce2rhoFresh", float(self.rhoIce) / float(self.rhoFresh)
        )
        object.__setattr__(
            self, "rhoFresh2rhoSnow", float(self.rhoFresh) / float(self.rhoSnow)
        )
        object.__setattr__(self, "lhSublim", float(self.lhFusion) + float(self.lhEvap))
        object.__setattr__(self, "recip_h0", 1.0 / float(self.h0))
        object.__setattr__(self, "recip_h0_south", 1.0 / float(self.h0_south))
        object.__setattr__(
            self, "sinWat", math.sin(math.radians(float(self.waterTurnAngle)))
        )
        object.__setattr__(
            self, "cosWat", math.cos(math.radians(float(self.waterTurnAngle)))
        )
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
