"""Registered experiment controls must describe the arrays actually allocated."""

import numpy as np
import pytest

from veris.configuration import SETTINGS, Configuration
from veris.initialization import initialize
from veris.physical_constants import PHYSICALCONSTANTS, PhysicalConstants
from veris.setups import artificial

EXPERIMENT_DEFAULTS = {
    "saltOcn_ref": 34.7,
    "artificialGridSpacing": 8000.0,
    "artificialWindSpeed": 5.0,
    "artificialAirTemperature": 260.0,
    "artificialIceThickness": 1.0,
    "artificialSnowThickness": 0.05,
    "artificialIceArea": 0.8,
    "artificialOceanDepth": -100.0,
    "artificialCoriolis": 1e-4,
    "artificialCooling": 100.0,
    "artificialTimeStep": 600.0,
    "artificialEVPsteps": 5,
}


def test_all_experiment_defaults_are_registered() -> None:
    assert EXPERIMENT_DEFAULTS.keys() == artificial.ARTIFICIAL_SETTINGS.keys()
    assert not EXPERIMENT_DEFAULTS.keys() & SETTINGS.keys()
    assert not EXPERIMENT_DEFAULTS.keys() & PHYSICALCONSTANTS.keys()
    settings = Configuration()
    for name, value in EXPERIMENT_DEFAULTS.items():
        assert artificial.ARTIFICIAL_SETTINGS[name].default == value
        assert not hasattr(settings, name)


def test_optical_snow_transition_is_a_physical_constant() -> None:
    assert "hCut" in PHYSICALCONSTANTS
    assert "hCut" not in SETTINGS
    assert PhysicalConstants().hCut == 0.15


def test_allocation_extents_are_recorded_and_settings_overrides_are_used() -> None:
    state, settings, _ = initialize(settings_overrides={"nx": 5, "ny": 6})
    assert (settings.nx, settings.ny) == (5, 6)
    assert state.hIceMean.shape == (9, 10)
    state, settings, _ = initialize(nx=4, settings_overrides={"nx": 5, "ny": 6})
    assert (settings.nx, settings.ny) == (4, 6)
    assert state.hIceMean.shape == (8, 10)


def test_artificial_overrides_drive_geometry_fields_and_time_controls() -> None:
    state, settings, constants = artificial.initialize(
        settings_overrides={"nx": 4, "ny": 5},
        scenario_overrides={
            "saltOcn_ref": 33,
            "artificialGridSpacing": 2000,
            "artificialWindSpeed": -3,
            "artificialAirTemperature": 265,
            "artificialIceThickness": 2,
            "artificialSnowThickness": 0.1,
            "artificialIceArea": 0.6,
            "artificialOceanDepth": -80,
            "artificialCoriolis": -1e-4,
            "artificialTimeStep": 300,
            "artificialEVPsteps": 7,
        },
        physical_overrides={"rhoIce": 910, "rhoSnow": 310},
    )
    assert (settings.nx, settings.ny) == (4, 5)
    np.testing.assert_array_equal(state.ocSalt, 33)
    assert (settings.deltatDyn, settings.deltatTherm, settings.nEVPsteps) == (
        300,
        300,
        7,
    )
    assert not settings.use_sharding
    mask = np.asarray(state.iceMask)
    np.testing.assert_array_equal(state.hIceMean, 2 * mask)
    np.testing.assert_array_equal(state.hSnowMean, 0.1 * mask)
    np.testing.assert_array_equal(state.Area, 0.6 * mask)
    np.testing.assert_array_equal(
        state.SeaIceLoad, (2 * constants.rhoIce + 0.1 * constants.rhoSnow) * mask
    )
    np.testing.assert_array_equal(state.dxG, 2000)
    np.testing.assert_array_equal(state.recip_dxC, 1 / 2000)
    np.testing.assert_array_equal(state.rAz, 2000**2)
    np.testing.assert_array_equal(state.ATemp, 265)
    np.testing.assert_array_equal(state.TSurf, 265)
    np.testing.assert_array_equal(state.uWind, -3)
    np.testing.assert_array_equal(state.wSpeed, 3)
    np.testing.assert_array_equal(state.R_low, -80)
    np.testing.assert_array_equal(state.fCori, -1e-4)


def test_explicit_artificial_arguments_and_model_timestep_overrides_win() -> None:
    state, settings, _ = artificial.initialize(
        nx=4,
        ny=6,
        wind=2,
        air_temperature=262,
        settings_overrides={
            "nx": 8,
            "ny": 8,
            "deltatDyn": 200,
            "deltatTherm": 400,
            "nEVPsteps": 3,
        },
        scenario_overrides={"artificialWindSpeed": 4, "artificialAirTemperature": 266},
    )
    assert (settings.nx, settings.ny) == (4, 6)
    np.testing.assert_array_equal(state.uWind, 2)
    np.testing.assert_array_equal(state.ATemp, 262)
    assert (settings.deltatDyn, settings.deltatTherm, settings.nEVPsteps) == (
        200,
        400,
        3,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"artificialGridSpacing": 0},
        {"artificialAirTemperature": 0},
        {"artificialIceThickness": -1},
        {"artificialSnowThickness": -1},
        {"artificialIceArea": 1.1},
        {"artificialOceanDepth": 10},
        {"artificialTimeStep": 0},
        {"artificialEVPsteps": 0},
        {"artificialEVPsteps": 2.5},
    ],
)
def test_invalid_registered_experiment_controls_are_rejected(
    kwargs: dict[str, float | int | bool],
) -> None:
    assert kwargs.keys() <= artificial.ARTIFICIAL_SETTINGS.keys()
    with pytest.raises((ValueError, TypeError), match=next(iter(kwargs))):
        artificial.initialize(scenario_overrides=kwargs)


def test_artificial_rejects_sharding_without_mesh_support() -> None:
    with pytest.raises(ValueError, match="shard|serial"):
        artificial.initialize(settings_overrides={"use_sharding": True})


def test_omitted_cooling_uses_local_experiment_default() -> None:
    state, settings, constants = artificial.initialize(nx=4, ny=4)
    implicit = artificial.step(state, settings, constants)
    explicit = artificial.step(state, settings, constants, cooling=100.0)
    np.testing.assert_array_equal(implicit.hIceMean, explicit.hIceMean)
    np.testing.assert_array_equal(implicit.Qnet, explicit.Qnet)


def test_artificial_controls_cannot_enter_model_settings() -> None:
    with pytest.raises(TypeError, match="artificialWindSpeed"):
        initialize(settings_overrides={"artificialWindSpeed": 4})
    with pytest.raises(TypeError, match="unknownScenario"):
        artificial.initialize(scenario_overrides={"unknownScenario": 4})


def test_initializer_rejects_step_only_cooling_override() -> None:
    with pytest.raises(ValueError, match="cooling argument to step"):
        artificial.initialize(scenario_overrides={"artificialCooling": 25})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scenario_config_is_frozen_and_uses_model_precision(dtype: str) -> None:
    from dataclasses import FrozenInstanceError, replace

    scenario = artificial.ArtificialSettings(dtype=dtype)
    assert isinstance(scenario.artificialGridSpacing, np.dtype(dtype).type)
    with pytest.raises(FrozenInstanceError):
        scenario.artificialGridSpacing = 2  # ty: ignore[invalid-assignment]
    assert replace(scenario, artificialGridSpacing=2000).artificialGridSpacing == 2000
