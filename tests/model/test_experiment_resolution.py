"""Authored overlays must not promote inherited or rounded preset defaults."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from veris import Execution, Experiment, Forcing, Grid
from veris.io import OutputSettings, Stream


def test_scenario_replacement_recomputes_inherited_controls() -> None:
    """Replacing scenario controls must not replay the preceding preset values."""
    base = Experiment.builtin("island")
    changed = replace(
        base,
        scenario={"islandTimeStep": 300.0, "islandEVPsteps": 7, "islandCooling": 25.0},
    )
    assert changed.configuration.deltatDyn == 300.0
    assert changed.configuration.deltatTherm == 300.0
    assert changed.configuration.nEVPsteps == 7
    assert changed.resolved_forcing.at(0) == {"Qnet": 25.0, "Qsw": 0.0}


@pytest.mark.parametrize(
    "preset, name, expected",
    [("island", "saltOcn_ref", 34.7), ("growth", "hIceMean", 1.3)],
)
def test_dtype_promotion_recomputes_inherited_defaults(
    preset: str, name: str, expected: float
) -> None:
    """A precision overlay must evaluate original preset literals in its dtype."""
    low = Experiment.builtin(preset, settings={"dtype": "float32"})
    high = replace(low, settings={"dtype": "float64"})
    # The preceding implementation exposes only the rounded scenario mapping.
    scenario = getattr(high, "resolved_scenario", high.scenario)
    assert scenario[name] == expected


def test_explicit_settings_and_heat_continue_to_win_over_scenario() -> None:
    """Changing inherited defaults must preserve deliberately authored values."""
    base = Experiment.builtin(
        "island",
        settings={"deltatDyn": 30.0, "deltatTherm": 30.0, "nEVPsteps": 2},
        forcing=Forcing(constant={"Qnet": 70.0}),
    )
    changed = base.with_overrides(
        {
            "scenario": {
                "islandTimeStep": 300.0,
                "islandEVPsteps": 7,
                "islandCooling": 25.0,
            }
        }
    )
    assert changed.configuration.deltatDyn == 30.0
    assert changed.configuration.nEVPsteps == 2
    assert changed.resolved_forcing.at(0) == {"Qnet": 70.0, "Qsw": 0.0}


def test_raw_inputs_are_sparse_owned_and_distinct_from_resolution() -> None:
    """Mutating caller mappings must never alter current or future resolution."""
    settings = {"dtype": "float32"}
    scenario = {"islandWindSpeed": -3.125}
    definition = Experiment.builtin("island", settings=settings, scenario=scenario)
    settings["dtype"] = "float64"
    scenario["islandWindSpeed"] = 9.0
    assert dict(definition.settings) == {"dtype": "float32"}
    assert dict(definition.scenario) == {"islandWindSpeed": -3.125}
    assert not definition.forcing.constant
    assert definition.resolved_scenario["islandWindSpeed"] == -3.125
    assert definition.resolved_forcing.at(0)["Qnet"] == 100.0
    with pytest.raises(TypeError):
        definition.scenario["islandWindSpeed"] = 1.0  # ty: ignore[invalid-assignment]


def test_authored_rounded_scalar_stays_rounded_when_promoted() -> None:
    """An explicit rounded scalar is data, rather than an inherited default."""
    rounded = float(np.float32(1.3))
    definition = Experiment.builtin(
        "growth", settings={"dtype": "float32"}, scenario={"hIceMean": rounded}
    ).with_overrides({"settings": {"dtype": "float64"}})
    assert definition.resolved_scenario["hIceMean"] == rounded
    assert definition.scenario["hIceMean"] == rounded


def test_mapping_overlays_merge_and_arrays_replace_without_aliasing() -> None:
    """Nested explicit updates retain sibling entries and replace array values."""
    replacement = np.full((4, 6), 9.0)
    base = Experiment.builtin(
        "island",
        grid=Grid(4, 6),
        forcing=Forcing(constant={"uWind": np.ones((4, 6)), "ocSalt": 30.0}),
        execution=Execution(backend="cpu"),
        settings={"useRelativeWind": True},
        steps=2,
    )
    changed = base.with_overrides(
        {
            "forcing": {"constant": {"uWind": replacement}},
            "execution": {"partitions": [2, 1]},
            "settings": {"useRelativeWind": False},
            "steps": 0,
            "output": None,
        }
    )
    replacement[:] = 100.0
    np.testing.assert_array_equal(changed.resolved_forcing.constant["uWind"], 9.0)
    np.testing.assert_array_equal(base.resolved_forcing.constant["uWind"], 1.0)
    assert changed.resolved_forcing.constant["ocSalt"] == 30.0
    assert changed.execution == Execution(backend="cpu", partitions=(2, 1))
    assert not changed.configuration.useRelativeWind
    assert changed.steps == 0 and changed.output is None


def test_output_overlay_preserves_streams_and_allows_explicit_disable() -> None:
    """Updating one output key must not reset sampling or variable selection."""
    output = OutputSettings(
        streams=(Stream(variables=("Area",), sampling_interval=timedelta(seconds=600)),)
    )
    changed = Experiment.builtin("island", output=output).with_overrides(
        {"output": {"enabled": False}}
    )
    assert changed.output is not None and not changed.output.enabled
    assert changed.output.streams == output.streams


def test_file_configuration_and_python_overlay_follow_preset_precedence(
    tmp_path: Path,
) -> None:
    """Shared daily defaults, island preset and two explicit layers resolve once."""
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "preset": "island",
                "scenario": {"islandTimeStep": 300.0, "islandCooling": 25.0},
                "settings": {"deltatDyn": 120.0, "deltatTherm": 120.0},
                "constants": {"rhoIce": 910.0},
                "forcing": {"constant": {"Qnet": 70.0}},
                "steps": 2,
            }
        )
    )
    definition = Experiment.from_file(path).with_overrides(
        {
            "settings": {"deltatDyn": 30.0, "deltatTherm": 30.0},
            "constants": {"rhoIce": 905.0},
            "forcing": {"series": {"Qnet": [0.0, 110.0]}},
        }
    )
    assert definition.configuration.deltatDyn == 30.0
    assert definition.configuration.nEVPsteps == 5
    assert definition.physical_constants.rhoIce == 905.0
    assert definition.resolved_forcing.at(0)["Qnet"] == 0.0
    assert definition.resolved_forcing.at(1)["Qnet"] == 110.0


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"unknown": 1}, "unknown"),
        ({"settings": {"unknown": 1}}, "unknown"),
        ({"settings": {"nx": 4}}, "Grid"),
        ({"execution": {"unknown": 1}}, "unknown"),
        ({"scenario": {"unknown": 1}}, "unknown"),
        ({"forcing": {"constant": {"Area": 0.5}}}, "Area"),
        ({"settings": {"deltatDyn": 300.0}}, "coupled.*timestep"),
    ],
)
def test_invalid_overlay_is_rejected_without_mutating_definition(
    overrides: dict[str, Any], message: str
) -> None:
    base = Experiment.builtin("island")
    with pytest.raises((ValueError, TypeError), match=message):
        base.with_overrides(overrides)
    assert base.configuration.deltatDyn == 600.0
    assert base.resolved_forcing.at(0)["Qnet"] == 100.0


def test_repeated_overlays_do_not_accumulate_resolved_defaults() -> None:
    """A second overlay uses authored controls rather than a prior resolution."""
    base = Experiment.builtin("island")
    first = base.with_overrides({"scenario": {"islandTimeStep": 300.0}})
    second = first.with_overrides({"scenario": {"islandTimeStep": 150.0}})
    assert second.configuration.deltatDyn == 150.0
    assert second.configuration.deltatTherm == 150.0
    assert dict(second.settings) == {}
    assert second.with_overrides({}).to_dict() == second.to_dict()
