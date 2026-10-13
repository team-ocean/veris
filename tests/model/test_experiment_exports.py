"""Authored exports retain omission; resolved exports deliberately pin values."""

import json
from typing import Any

import numpy as np
import pytest

from veris import Experiment, Forcing, Grid
from veris.model.experiment_file import experiment_from_dict


@pytest.mark.parametrize("kind", ["island", "dynamics", "growth"])
def test_authored_export_remains_sparse_and_resolved_export_pins_values(
    kind: str,
) -> None:
    definition = Experiment.builtin(kind, settings={"dtype": "float32"})
    authored = definition.to_dict(mode="authored")
    resolved = definition.to_dict(mode="resolved")
    assert authored["preset"] == kind
    assert authored["settings"] == {"dtype": "float32"}
    assert not authored.get("scenario") and "grid" not in authored
    assert "deltatDyn" in resolved["settings"] and "rhoIce" in resolved["constants"]
    assert resolved["grid"] == {"nx": definition.grid.nx, "ny": definition.grid.ny}
    restored = experiment_from_dict(authored)
    assert restored.configuration == definition.configuration
    assert restored.to_dict(mode="authored") == authored
    json.dumps(authored, allow_nan=False)
    json.dumps(resolved, allow_nan=False)


@pytest.mark.parametrize(
    "kind,field,expected",
    [("island", "saltOcn_ref", 34.7), ("growth", "hIceMean", 1.3)],
)
def test_authored_precision_promotion_recomputes_defaults_and_resolved_keeps_literals(
    kind: str, field: str, expected: float
) -> None:
    definition = Experiment.builtin(kind, settings={"dtype": "float32"})
    authored = experiment_from_dict(definition.to_dict(mode="authored")).with_overrides(
        {"settings": {"dtype": "float64"}}
    )
    resolved = experiment_from_dict(definition.to_dict(mode="resolved")).with_overrides(
        {"settings": {"dtype": "float64"}, "constants": {"dtype": "float64"}}
    )
    assert authored.resolved_scenario[field] == expected
    assert resolved.resolved_scenario[field] == float(np.float32(expected))


def test_profile_identity_and_explicit_zero_survive_authored_roundtrip() -> None:
    definition = Experiment.builtin("dynamics", profile="example", steps=0)
    authored = definition.to_dict(mode="authored")
    assert authored["profile"] == "example" and authored["steps"] == 0
    assert "grid" not in authored
    restored = experiment_from_dict(authored).with_overrides({"profile": "standalone"})
    assert restored.grid == Grid(1024, 1024) and restored.steps == 0
    assert definition.to_dict()["profile"] == "example"


def test_partial_grouped_output_export_preserves_authored_omissions() -> None:
    definition = experiment_from_dict(
        {"preset": "growth", "output": {"enabled": False}}
    )
    assert definition.to_dict(mode="authored")["output"] == {"enabled": False}
    assert "streams" in definition.to_dict(mode="resolved")["output"]


@pytest.mark.parametrize("mode", ["unknown", "", None, False])
def test_unknown_export_mode_has_informative_error(mode: Any) -> None:
    with pytest.raises((ValueError, TypeError), match="mode"):
        Experiment.builtin("growth").to_dict(mode=mode)


def test_authored_arrays_are_fresh_and_round_trip_without_promoting_defaults() -> None:
    original = np.arange(24, dtype=float).reshape(4, 6)
    definition = Experiment.builtin(
        "dynamics", grid=Grid(4, 6), forcing=Forcing(constant={"uWind": original})
    )
    first = definition.to_dict(mode="authored")
    restored = experiment_from_dict(first)
    np.testing.assert_array_equal(restored.resolved_forcing.constant["uWind"], original)
    first["forcing"]["constant"]["uWind"][0][0] = 1000.0
    assert (
        definition.to_dict(mode="authored")["forcing"]["constant"]["uWind"][0][0] == 0.0
    )
    assert not restored.forcing.constant["uWind"].flags.writeable


def test_new_profile_representation_uses_version_two_and_version_one_still_loads() -> (
    None
):
    definition = Experiment.builtin("growth", profile="example")
    assert definition.to_dict()["schema_version"] == 2
    restored = experiment_from_dict(
        {"schema_version": 2, "preset": "growth", "profile": "example"}
    )
    assert restored.profile == "example"
    legacy = experiment_from_dict({"schema_version": 1, "preset": "growth"})
    assert legacy.preset == "growth" and legacy.profile is None


def test_partial_profile_execution_and_output_controls_preserve_other_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from veris.setups import presets

    base = presets.get_preset("growth")
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    presets.register_preset(
        replace(
            base,
            name="profiled_column",
            profiles={
                "cluster": {
                    "grid": (4, 4),
                    "execution": {
                        "backend": "cpu",
                        "partitions": (1, 2),
                        "distributed": {
                            "coordinator_address": "host:12345",
                            "num_processes": 2,
                            "process_id": 0,
                        },
                    },
                    "output": {
                        "calendar": "360_day",
                        "enabled": True,
                        "streams": [
                            {
                                "variables": ["hIceMean"],
                                "sampling_interval": {"seconds": 86400},
                            }
                        ],
                    },
                }
            },
        )
    )
    authored = {
        "preset": "profiled_column",
        "profile": "cluster",
        "execution": {"distributed": {"process_id": 1}},
        "output": {"enabled": False},
    }
    definition = experiment_from_dict(authored)
    assert definition.execution.distributed is not None
    assert definition.execution.distributed.process_id == 1
    assert definition.execution.distributed.coordinator_address == "host:12345"
    assert definition.output is not None and definition.output.calendar == "360_day"
    assert not definition.output.enabled and definition.output.streams[0].variables == (
        "hIceMean",
    )
    exported = definition.to_dict(mode="authored")
    assert exported["execution"] == authored["execution"]
    assert exported["output"] == authored["output"]
