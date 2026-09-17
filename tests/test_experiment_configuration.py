"""Registered experiment controls must describe the arrays actually allocated."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from veris.configuration import SETTINGS, Configuration
from veris.initialization import initialize
from veris.physical_constants import PHYSICALCONSTANTS
from veris.setups import island

EXPERIMENT_DEFAULTS = {
    "saltOcn_ref": 34.7,
    "islandGridSpacing": 8000.0,
    "islandWindSpeed": 5.0,
    "islandAirTemperature": 260.0,
    "islandIceThickness": 1.0,
    "islandSnowThickness": 0.05,
    "islandIceArea": 0.8,
    "islandOceanDepth": -100.0,
    "islandCoriolis": 1e-4,
    "islandCooling": 100.0,
    "islandTimeStep": 600.0,
    "islandEVPsteps": 5,
}


def test_all_experiment_defaults_are_registered() -> None:
    assert EXPERIMENT_DEFAULTS.keys() == island.ISLAND_SETTINGS.keys()
    assert not EXPERIMENT_DEFAULTS.keys() & SETTINGS.keys()
    assert not EXPERIMENT_DEFAULTS.keys() & PHYSICALCONSTANTS.keys()
    scenario = island.IslandSettings()
    with pytest.raises(FrozenInstanceError):
        scenario.islandGridSpacing = 2  # ty: ignore[invalid-assignment]
    assert replace(scenario, islandGridSpacing=2000).islandGridSpacing == 2000
    settings = Configuration()
    for name, value in EXPERIMENT_DEFAULTS.items():
        assert island.ISLAND_SETTINGS[name].default == value
        assert not hasattr(settings, name)


def test_allocation_extents_are_recorded_and_settings_overrides_are_used() -> None:
    state, settings, _ = initialize(settings_overrides={"nx": 5, "ny": 6})
    assert (settings.nx, settings.ny) == (5, 6)
    assert state.hIceMean.shape == (9, 10)
    state, settings, _ = initialize(nx=4, settings_overrides={"nx": 5, "ny": 6})
    assert (settings.nx, settings.ny) == (4, 6)
    assert state.hIceMean.shape == (8, 10)


def test_island_overrides_drive_geometry_fields_and_time_controls() -> None:
    state, settings, constants = island.initialize(
        settings_overrides={"nx": 4, "ny": 5},
        scenario_overrides={
            "saltOcn_ref": 33,
            "islandGridSpacing": 2000,
            "islandWindSpeed": -3,
            "islandAirTemperature": 265,
            "islandIceThickness": 2,
            "islandSnowThickness": 0.1,
            "islandIceArea": 0.6,
            "islandOceanDepth": -80,
            "islandCoriolis": -1e-4,
            "islandTimeStep": 300,
            "islandEVPsteps": 7,
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


def test_explicit_island_arguments_and_model_timestep_overrides_win() -> None:
    state, settings, _ = island.initialize(
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
        scenario_overrides={"islandWindSpeed": 4, "islandAirTemperature": 266},
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
        {"islandGridSpacing": 0},
        {"islandAirTemperature": 0},
        {"islandIceThickness": -1},
        {"islandSnowThickness": -1},
        {"islandIceArea": 1.1},
        {"islandOceanDepth": 10},
        {"islandTimeStep": 0},
        {"islandEVPsteps": 0},
        {"islandEVPsteps": 2.5},
    ],
)
def test_invalid_registered_experiment_controls_are_rejected(
    kwargs: dict[str, float | int | bool],
) -> None:
    assert kwargs.keys() <= island.ISLAND_SETTINGS.keys()
    with pytest.raises((ValueError, TypeError), match=next(iter(kwargs))):
        island.initialize(scenario_overrides=kwargs)


def test_island_rejects_sharding_without_mesh_support() -> None:
    with pytest.raises(ValueError, match="shard|serial"):
        island.initialize(settings_overrides={"use_sharding": True})


def test_island_controls_cannot_enter_model_settings() -> None:
    with pytest.raises(TypeError, match="islandWindSpeed"):
        initialize(settings_overrides={"islandWindSpeed": 4})
    with pytest.raises(TypeError, match="unknownScenario"):
        island.initialize(scenario_overrides={"unknownScenario": 4})


def test_initializer_rejects_step_only_cooling_override() -> None:
    with pytest.raises(ValueError, match="cooling argument to step"):
        island.initialize(scenario_overrides={"islandCooling": 25})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scenario_config_uses_model_precision(dtype: str) -> None:
    scenario = island.IslandSettings(dtype=dtype)
    assert isinstance(scenario.islandGridSpacing, np.dtype(dtype).type)
