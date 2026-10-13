"""Preset definitions share authoritative scalar controls with standalone setups."""

from dataclasses import FrozenInstanceError

import pytest

from veris import Experiment, Forcing, Grid, initialize_experiment
from veris.setups.island import IslandSettings
from veris.setups.run_dyn import DynamicsSettings
from veris.setups.run_growth import GrowthSettings


def test_builtin_numerical_defaults_and_explicit_precedence() -> None:
    island = Experiment.builtin("island")
    scenario = IslandSettings()
    assert island.configuration.deltatDyn == scenario.islandTimeStep
    assert island.configuration.deltatTherm == scenario.islandTimeStep
    assert island.configuration.nEVPsteps == scenario.islandEVPsteps
    custom = Experiment.builtin(
        "island",
        settings={"deltatDyn": 30.0, "deltatTherm": 30.0, "nEVPsteps": 2},
        scenario={"islandTimeStep": 60.0, "islandEVPsteps": 3},
    )
    assert custom.configuration.deltatDyn == 30.0
    assert custom.configuration.nEVPsteps == 2


def test_preset_scenario_defaults_and_overrides_are_reproducible_and_immutable() -> (
    None
):
    controls = {"length": 12000.0, "wind_speed": -3.0}
    definition = Experiment.builtin("dynamics", scenario=controls)
    controls["length"] = 20000.0
    assert definition.scenario["length"] == 12000.0
    assert definition.scenario["wind_speed"] == -3.0
    assert definition.resolved_scenario["wind_decay"] == DynamicsSettings().wind_decay
    with pytest.raises(TypeError):
        definition.scenario["length"] = 20000.0  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        definition.preset = "island"  # ty: ignore[invalid-assignment]


def test_growth_grid_and_scenario_do_not_duplicate_experiment_controls() -> None:
    definition = Experiment.builtin("growth")
    controls = GrowthSettings()
    assert definition.grid == Grid(2, 2)
    assert definition.resolved_scenario["hIceMean"] == controls.hIceMean
    assert not {"nx", "ny", "steps", "dtype"} & definition.resolved_scenario.keys()


def test_growth_uses_shared_numerical_defaults_and_explicit_overrides() -> None:
    """Use shared physics controls with daily steps and five thickness categories."""
    from veris.model.configuration import Configuration

    expected = Configuration(nx=2, ny=2, use_sharding=False)
    assert Experiment.builtin("growth").configuration == expected
    assert initialize_experiment(Experiment.builtin("growth"))[1] == expected
    options = {"deltatTherm": 600.0, "deltatDyn": 600.0, "nITC": 1}
    conf = Experiment.builtin("growth", settings=options).configuration
    assert conf.deltatTherm == 600.0
    assert conf.deltatDyn == 600.0
    assert conf.nITC == 1
    assert (
        initialize_experiment(Experiment.builtin("growth", settings=options))[1] == conf
    )


def test_explicit_series_override_prescribed_preset_heat() -> None:
    definition = Experiment.builtin(
        "island",
        forcing=Forcing(series={"Qnet": [70.0, 110.0]}),
        steps=2,
    )
    assert definition.resolved_forcing.at(0)["Qnet"] == 70.0
    assert definition.resolved_forcing.at(1)["Qnet"] == 110.0
    assert definition.resolved_forcing.at(1)["Qsw"] == 0.0


@pytest.mark.parametrize(
    "name,scenario",
    [
        ("island", {"unknown": 1}),
        ("dynamics", {"length": -1.0}),
        ("growth", {"nx": 5}),
        ("growth", {"hIceMean": -1.0}),
    ],
)
def test_invalid_preset_controls_fail_before_allocation(
    name: str, scenario: dict[str, float]
) -> None:
    with pytest.raises((ValueError, TypeError), match="unknown|length|Grid|hIceMean"):
        Experiment.builtin(name, scenario=scenario)


@pytest.mark.parametrize("name", ["nx", "ny", "steps"])
def test_growth_scenario_rejects_experiment_run_controls(name: str) -> None:
    """Run controls belong to Grid/Experiment and cannot be silently ignored."""
    with pytest.raises(TypeError, match=name):
        GrowthSettings(**{name: 4})  # ty: ignore[invalid-argument-type]


def test_growth_run_controls_have_one_canonical_owner() -> None:
    """Profile defaults and explicit run controls reach the allocated trajectory."""
    definition = Experiment.builtin("growth", profile="standalone")
    assert definition.grid == Grid(2, 2) and definition.steps == 150
    custom = definition.with_overrides({"grid": {"nx": 4, "ny": 6}, "steps": 2})
    state, _, _ = initialize_experiment(custom)
    assert state.hIceMean.shape == (8, 10) and custom.steps == 2
