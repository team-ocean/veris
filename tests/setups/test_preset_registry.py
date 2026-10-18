"""Presets supply controls and field factories through one extensible contract."""

import json
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from veris import Experiment, Grid
from veris.io import OutputSettings, Stream
from veris.model.configuration import SETTINGS, Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import Parameter, State
from veris.setups import presets, run_growth


@pytest.mark.parametrize(
    "name,profile,shape,steps,seconds,evp",
    [
        ("island", None, (8, 12), 0, 600.0, 5),
        ("island", "example", (8, 12), 3, 600.0, 5),
        ("dynamics", None, (8, 12), 0, 600.0, 120),
        ("dynamics", "example", (16, 24), 100, 600.0, 120),
        ("dynamics", "standalone", (1024, 1024), 100, 600.0, 120),
        ("dynamics", "parallel", (1024, 1024), 1000, 600.0, 120),
        ("growth", None, (2, 2), 0, 86400.0, None),
        ("growth", "example", (2, 2), 150, 86400.0, None),
        ("growth", "standalone", (2, 2), 150, 86400.0, None),
    ],
)
def test_profiles_preserve_existing_entry_defaults(
    name: str,
    profile: str | None,
    shape: tuple[int, int],
    steps: int,
    seconds: float,
    evp: int | None,
) -> None:
    definition = Experiment.builtin(name, profile=profile)
    assert definition.grid == Grid(*shape)
    assert definition.steps == steps
    assert definition.step_duration.total_seconds() == seconds
    if evp is not None:
        assert definition.configuration.nEVPsteps == evp


def test_explicit_profile_overrides_and_later_profile_changes_preserve_authorship() -> (
    None
):
    base = Experiment.builtin("dynamics", profile="example")
    changed = base.with_overrides({"profile": "parallel"})
    assert changed.grid == Grid(1024, 1024) and changed.steps == 1000
    explicit = Experiment.builtin(
        "dynamics", profile="example", grid=Grid(4, 6), steps=0
    ).with_overrides({"profile": "parallel"})
    assert explicit.grid == Grid(4, 6) and explicit.steps == 0


def _controls(values: Mapping[str, Any], dtype: str) -> dict[str, Any]:
    """A future setup's validated scalar schema has no core setup-name branch."""
    return {"column_ice": float(values.get("column_ice", 0.6))}


def _column_fields(
    state: State,
    configuration: Configuration,
    constants: PhysicalConstants,
    scenario: Mapping[str, Any],
    global_shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    *,
    regional: bool,
) -> dict[str, Any]:
    return {"hIceMean": jnp.full_like(state.hIceMean, scenario["column_ice"])}


def test_additional_registered_setup_needs_no_core_name_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presets.get_preset("growth")
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    descriptor = presets.SetupPreset(
        name="test_column",
        physics="thermodynamics",
        scenario_registry={"column_ice": Parameter(0.6, float, "Initial ice", "m")},
        scenario_factory=_controls,
        field_factory=_column_fields,
        grid=(4, 6),
        numerical_defaults={"deltatTherm": 60.0},
    )
    presets.register_preset(descriptor)
    definition = Experiment.builtin("test_column", scenario={"column_ice": 0.8})
    assert definition.grid == Grid(4, 6)
    assert definition.physics == "thermodynamics"
    assert definition.configuration.deltatTherm == 60.0
    assert definition.resolved_scenario == {"column_ice": 0.8}
    with pytest.raises(ValueError, match="duplicate|already"):
        presets.register_preset(descriptor)
    with pytest.raises(ValueError, match="unknown"):
        Experiment.builtin("test_column", scenario={"typo": 1.0})


@pytest.mark.parametrize("name", ["island", "dynamics", "growth"])
def test_unknown_profile_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="profile"):
        Experiment.builtin(name, profile="unknown")


def test_growth_numerical_choices_reference_the_shared_owner() -> None:
    assert run_growth.GROWTH_NUMERICAL_DEFAULTS == {
        name: SETTINGS[name].default for name in ("deltatTherm", "deltatDyn", "nITC")
    }
    # Registration exposes the same intended choices and excludes run controls
    # from the scientific scenario result.
    descriptor = presets.get_preset("growth")
    assert descriptor.grid == (2, 2)
    assert (
        not {"nx", "ny", "steps", "dtype"}
        & Experiment.builtin("growth").resolved_scenario.keys()
    )


def test_profile_physical_initial_and_output_defaults_share_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presets.get_preset("growth")
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    output = OutputSettings(
        streams=(Stream(variables=("Area",), sampling_interval=timedelta(seconds=600)),)
    )
    descriptor = presets.SetupPreset(
        name="profile_column",
        physics="thermodynamics",
        scenario_registry={"column_ice": Parameter(0.6, float, "Initial ice", "m")},
        scenario_factory=_controls,
        field_factory=_column_fields,
        grid=(4, 6),
        profiles={
            "example": {
                "constants": {"rhoIce": 905.0},
                "initial_conditions": {"hIceMean": 0.75},
                "output": output,
            }
        },
    )
    presets.register_preset(descriptor)
    base = Experiment.builtin("profile_column", profile="example")
    assert base.physical_constants.rhoIce == 905.0
    assert base.resolved_initial_conditions["hIceMean"] == 0.75
    assert base.output == output
    changed = base.with_overrides(
        {
            "constants": {"rhoIce": 900.0},
            "initial_conditions": {"hIceMean": 0.8},
            "output": None,
        }
    )
    assert changed.physical_constants.rhoIce == 900.0
    assert changed.resolved_initial_conditions["hIceMean"] == 0.8
    assert changed.output is None


def test_profile_unknown_controls_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    presets.get_preset("growth")
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    with pytest.raises(ValueError, match="unknown.*profile|profile.*unknown"):
        descriptor = presets.SetupPreset(
            name="bad_profile",
            physics="thermodynamics",
            scenario_registry={},
            scenario_factory=_controls,
            field_factory=_column_fields,
            profiles={"example": {"unknown": 1}},
        )
        presets.register_preset(descriptor)
        Experiment.builtin("bad_profile", profile="example")


def test_partial_declarative_grid_inherits_preset_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "column.json"
    path.write_text(json.dumps({"preset": "growth", "grid": {"nx": 4}}))
    definition = Experiment.from_file(path)
    assert definition.grid == Grid(4, 2)


def test_partition_override_is_owned_across_later_resolution() -> None:
    partitions = [2, 1]
    definition = Experiment.builtin("island", grid=Grid(4, 6)).with_overrides(
        {"execution": {"partitions": partitions}}
    )
    partitions[0] = 1
    assert definition.with_overrides({}).execution.partitions == (2, 1)


def test_profile_array_defaults_are_owned_after_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presets.get_preset("growth")
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    initial = np.full((4, 6), 0.75)
    descriptor = presets.SetupPreset(
        name="owned_profile",
        physics="thermodynamics",
        scenario_registry={"column_ice": Parameter(0.6, float, "Initial ice", "m")},
        scenario_factory=_controls,
        field_factory=_column_fields,
        grid=(4, 6),
        profiles={"example": {"initial_conditions": {"hIceMean": initial}}},
    )
    initial[:] = 0.9
    presets.register_preset(descriptor)
    experiment = Experiment.builtin("owned_profile", profile="example")
    np.testing.assert_array_equal(
        experiment.resolved_initial_conditions["hIceMean"], 0.75
    )


def test_registry_retains_parameter_metadata_for_inspection() -> None:
    descriptor = presets.SetupPreset(
        name="metadata_column",
        physics="thermodynamics",
        scenario_registry={"column_ice": Parameter(0.6, float, "Initial ice", "m")},
        scenario_factory=_controls,
        field_factory=_column_fields,
    )
    metadata = descriptor.scenario_registry["column_ice"]
    assert metadata.default == 0.6 and metadata.type is float
    assert metadata.description == "Initial ice" and metadata.units == "m"
