"""Independent defaults and host-validation contracts for extracted kernel laws."""

from dataclasses import replace

import pytest

from veris.configuration import SETTINGS, Settings
from veris.physical_constants import PHYSICALCONSTANTS, PhysicalConstants

PHYSICAL_DEFAULTS = {
    "iceVaporPressureTemperature": 2663.5,
    "iceVaporPressureLog10Offset": 12.537,
    "waterVaporDryAirMassRatio": 0.622,
    "iceSurfacePressure": 100000.0,
    "iceShortwaveExtinction": 1.5,
    "McPheeTaperArea": 0.4,
    "McPheeTaperSteepness": 7.0,
    "lateralMeltAreaFactor": 0.5,
    "cesmSaturationHumidityScale": 640380.0,
    "cesmSaturationHumidityTemperature": 5107.4,
    "augustVaporPressureLog10Offset": 9.4051,
    "augustVaporPressureTemperature": 2353.0,
    "mmHgToPa": 133.322,
    "neutralDragInverseWind": 0.0027,
    "neutralDragConstant": 0.000142,
    "neutralDragLinearWind": 0.0000764,
    "cesmUnstableMomentumOffset": 1.571,
    "longwaveHumidityPressureScale": 1000.0,
    "longwaveClearSkyOffset": 0.39,
    "longwaveHumidityCoefficient": 0.05,
    "seawaterHumidityFactor": 0.98,
    "cesmNeutralHeatUnstable": 0.0327,
    "cesmNeutralHeatStable": 0.018,
    "cesmNeutralMoisture": 0.0346,
    "bulkUnstableStabilityCoefficient": 16.0,
    "bulkStableStabilityCoefficient": 5.0,
    "lanlSaturationHumidityScale": 3.797915,
    "lanlSaturationExponentOffset": 7.93252e-6,
    "lanlSaturationExponentTemperature": 2.166847e-3,
    "lanlReferencePressure": 1013.0,
}
SETTING_DEFAULTS = {
    "surfaceTemperatureIterations": 6,
    "minActualIceThickness": 0.05,
    "basalDragSmoothing": 10.0,
    "basalDragMinArea": 0.01,
    "aEVPmassMin": 1e-4,
    "aEVPcStar": 4.0,
    "evpStressRelaxation": 1.0,
    "evpShearRelaxation": 0.25,
    "bulkStabilityLimit": 10.0,
    "lanlMinWindSpeed": 1.0,
    "lanlBulkIterations": 5,
}
CLOUD_LATITUDES = (
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
)
CLOUD_COEFFICIENTS = (
    0.88,
    0.84,
    0.80,
    0.76,
    0.72,
    0.68,
    0.63,
    0.59,
    0.52,
    0.50,
    0.50,
    0.50,
    0.52,
    0.59,
    0.63,
    0.68,
    0.72,
    0.76,
    0.80,
    0.84,
    0.88,
)


def test_scattered_coefficient_defaults_match_original_literals() -> None:
    for cls, registry, expected in (
        (Settings, SETTINGS, SETTING_DEFAULTS),
        (PhysicalConstants, PHYSICALCONSTANTS, PHYSICAL_DEFAULTS),
    ):
        obj = cls()
        assert expected.keys() <= registry.keys()
        for name, value in expected.items():
            assert registry[name].default == value, name
            assert getattr(obj, name) == value, name
            assert registry[name].description
    assert "cesmBulkIterations" not in SETTINGS
    assert "grav" not in PHYSICALCONSTANTS


def test_cloud_tables_are_immutable_hashable_and_replaceable() -> None:
    constants = PhysicalConstants()
    assert hasattr(constants, "longwaveCloudLatitudes")
    assert constants.longwaveCloudLatitudes == CLOUD_LATITUDES
    assert constants.longwaveCloudCoefficients == CLOUD_COEFFICIENTS
    assert hash(constants) == hash(PhysicalConstants())
    updated = replace(
        constants,
        longwaveCloudLatitudes=(-90, 0, 90),
        longwaveCloudCoefficients=(0.9, 0.5, 0.9),
    )
    assert updated.longwaveCloudLatitudes == (-90.0, 0.0, 90.0)
    assert hash(updated) != hash(constants)
    with pytest.raises(TypeError):
        updated.longwaveCloudLatitudes[0] = -80  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"longwaveCloudLatitudes": (-90, 90)},
        {"longwaveCloudLatitudes": (), "longwaveCloudCoefficients": ()},
        {"longwaveCloudLatitudes": (0,), "longwaveCloudCoefficients": (0.5,)},
        {"longwaveCloudLatitudes": (0, -90), "longwaveCloudCoefficients": (0.5, 0.9)},
        {"longwaveCloudLatitudes": (0, 0), "longwaveCloudCoefficients": (0.5, 0.5)},
        {
            "longwaveCloudLatitudes": (-90, float("inf")),
            "longwaveCloudCoefficients": (0.5, 0.5),
        },
        {
            "longwaveCloudLatitudes": (-90, 90),
            "longwaveCloudCoefficients": (0.5, float("nan")),
        },
        {"longwaveCloudLatitudes": [-90, 90]},
        {"longwaveCloudCoefficients": (True,) * 21},
        {"longwaveCloudCoefficients": ("0.5",) * 21},
    ],
)
def test_invalid_cloud_tables_are_rejected(kwargs: dict[str, object]) -> None:
    assert "longwaveCloudLatitudes" in PHYSICALCONSTANTS
    with pytest.raises((TypeError, ValueError), match="longwaveCloud"):
        PhysicalConstants(**kwargs)  # ty: ignore[invalid-argument-type] - intentionally invalid inputs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"surfaceTemperatureIterations": 0},
        {"surfaceTemperatureIterations": 1.5},
        {"lanlBulkIterations": -1},
        {"lanlBulkIterations": True},
        {"minActualIceThickness": 0},
        {"basalDragSmoothing": 0},
        {"aEVPmassMin": -1},
        {"lanlMinWindSpeed": 0},
    ],
)
def test_invalid_extracted_settings_are_rejected(kwargs: dict[str, object]) -> None:
    assert kwargs.keys() <= SETTINGS.keys()
    with pytest.raises((TypeError, ValueError), match=next(iter(kwargs))):
        Settings(**kwargs)  # ty: ignore[invalid-argument-type] - intentionally invalid inputs


def test_independent_relaxation_and_pressure_defaults_remain_independent() -> None:
    assert "evpShearRelaxation" in SETTINGS
    settings = replace(Settings(), evpStressRelaxation=2)
    assert settings.evpShearRelaxation == 0.25
    constants = replace(PhysicalConstants(), p0=101000, latvap=2502000)
    assert constants.iceSurfacePressure == 100000
    assert constants.lanlReferencePressure == 1013
    assert constants.lhEvap == 2500000
    with pytest.raises(TypeError, match="unknownCoefficient"):
        PhysicalConstants(unknownCoefficient=1)  # ty: ignore[unknown-argument]
