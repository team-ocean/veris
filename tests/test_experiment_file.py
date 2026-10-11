"""Declarative experiments preserve units, grouped overrides and resolved data."""

# ruff: noqa: DTZ001 -- model calendar dates are explicitly timezone-naive.

import json
from datetime import datetime, timedelta
from decimal import localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import veris
from veris.io import FixedDate, OutputSettings, Stream


def test_json_file_python_override_precedence_and_resolved_roundtrip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "grid": {"nx": 2, "ny": 4},
                "physics": "thermodynamics",
                "settings": {"dtype": "float32", "deltatTherm": 300, "nEVPsteps": 3},
                "constants": {"rhoIce": 910},
                "initial_conditions": {"hIceMean": 1.0, "Area": 0.8},
                "forcing": {
                    "constant": {"ocSalt": 34, "uWind": 2},
                    "series": {"Qnet": [70, 110, 85]},
                },
                "execution": {"backend": "cpu"},
                "steps": 3,
            }
        )
    )
    definition = veris.Experiment.from_file(
        path,
        overrides={
            "settings": {"nEVPsteps": 7},
            "forcing": {"constant": {"uWind": 4}, "series": {"Qnet": [80, 100, 120]}},
        },
    )
    assert definition.configuration.nEVPsteps == 7
    assert definition.configuration.deltatTherm == 300
    assert definition.physical_constants.rhoIce == 910
    assert definition.forcing.at(1)["ocSalt"] == 34
    assert definition.forcing.at(1)["uWind"] == 4
    assert definition.forcing.at(1)["Qnet"] == 100
    assert definition.initial_conditions["Area"] == 0.8
    resolved = definition.to_dict()
    assert resolved["schema_version"] == 1
    assert resolved["settings"]["nEVPsteps"] == 7
    assert resolved["settings"]["useEVP"] is True
    assert resolved["constants"]["rhoSnow"] == 330.0
    assert (
        not {"nx", "ny", "use_sharding", "recip_deltatDyn"}
        & resolved["settings"].keys()
    )
    encoded = json.dumps(resolved, allow_nan=False)
    path.write_text(encoded)
    restored = veris.Experiment.from_file(path)
    assert restored.configuration == definition.configuration
    assert restored.physical_constants == definition.physical_constants
    assert restored.to_dict() == resolved


@pytest.mark.parametrize("fixed", [False, True])
def test_output_calendar_and_microsecond_durations_roundtrip(
    tmp_path: Path, fixed: bool
) -> None:
    start = (
        FixedDate(2000, 2, 30, microsecond=123)
        if fixed
        else datetime(2000, 2, 29, microsecond=123)
    )
    output = OutputSettings(
        start=start,
        calendar="360_day" if fixed else "gregorian",
        streams=(
            Stream(
                name="ice",
                variables=("hIceMean", "Area"),
                sampling_interval=timedelta(microseconds=125000),
                period=timedelta(seconds=3, microseconds=1),
            ),
        ),
        sample_initial=False,
    )
    definition = veris.Experiment(output=output)
    path = tmp_path / "output.json"
    path.write_text(json.dumps(definition.to_dict(), allow_nan=False))
    restored = veris.Experiment.from_file(path)
    assert restored.output == output
    assert restored.to_dict() == definition.to_dict()


def test_human_duration_seconds_and_default_output_settings_are_supported(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output.json"
    path.write_text(
        json.dumps(
            {
                "output": {
                    "streams": [
                        {
                            "name": "ice",
                            "variables": ["hIceMean"],
                            "sampling_interval": {"seconds": 600},
                            "period": "daily",
                        }
                    ]
                },
            }
        )
    )
    definition = veris.Experiment.from_file(path)
    assert definition.output is not None
    assert definition.output.start == datetime(2000, 1, 1)
    stream = definition.output.streams[0]
    assert stream.sampling_interval == timedelta(minutes=10)
    assert stream.period == "daily"


def test_file_arrays_replace_whole_fields_and_cannot_alias_serialized_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "arrays.json"
    path.write_text(
        json.dumps(
            {
                "grid": {"nx": 2, "ny": 2},
                "initial_conditions": {"hIceMean": [[1, 2], [3, 4]], "Area": 0.8},
                "forcing": {"series": {"Qnet": [70, 110]}},
                "steps": 2,
            }
        )
    )
    definition = veris.Experiment.from_file(
        path,
        overrides={
            "initial_conditions": {"hIceMean": [[4, 3], [2, 1]]},
        },
    )
    np.testing.assert_array_equal(
        definition.initial_conditions["hIceMean"], [[4, 3], [2, 1]]
    )
    assert definition.initial_conditions["Area"] == 0.8
    serialized = definition.to_dict()
    serialized["initial_conditions"]["hIceMean"][0][0] = 999
    assert np.asarray(definition.initial_conditions["hIceMean"])[0, 0] == 4


@pytest.mark.parametrize(
    "data, message",
    [
        ([], "object|mapping"),
        ({"schema_version": 2}, "version"),
        ({"schema_version": True}, "version"),
        ({"unknown": 1}, "unknown"),
        ({"grid": {"nx": 2, "ny": 4, "unknown": 1}}, "unknown"),
        ({"settings": []}, "settings"),
        ({"forcing": {"constant": {}, "unknown": 1}}, "unknown"),
        ({"execution": {"partitions": [1]}}, "partition"),
        ({"execution": {"backend": "tpu"}}, "backend"),
        ({"output": {"streams": []}}, "Stream|stream"),
        (
            {"output": {"streams": [{"sampling_interval": {"seconds": 1e-8}}]}},
            "microsecond",
        ),
        (
            {"output": {"streams": [{"sampling_interval": {"seconds": True}}]}},
            "duration|seconds",
        ),
        (
            {
                "output": {
                    "streams": [
                        {"sampling_interval": {"seconds": 1, "microseconds": 1}}
                    ]
                }
            },
            "duration",
        ),
        ({"output": {"streams": [{"sampling_interval": {"unknown": 1}}]}}, "duration"),
        ({"output": {"unknown": 1}}, "unknown"),
        ({"initial_conditions": {"Area": 2}}, "Area"),
    ],
)
def test_invalid_declarative_input_is_rejected_with_context(
    tmp_path: Path, data: Any, message: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError), match=message):
        veris.Experiment.from_file(path)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"settings": {"nEVPsteps": 3, "nEVPsteps": 7}}')
    with pytest.raises(ValueError, match="duplicate.*nEVPsteps"):
        veris.Experiment.from_file(path)


def test_malformed_json_reports_input_path(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"grid":')
    with pytest.raises(ValueError, match="broken.json"):
        veris.Experiment.from_file(path)


def test_custom_geometry_and_builtin_scenario_roundtrip(tmp_path: Path) -> None:
    """Both user-defined geometry and preset controls survive resolved JSON."""
    geometry = veris.OceanGeometry(
        maskT=np.ones((2, 3, 1)),
        maskU=np.ones((2, 3, 1)),
        maskV=np.ones((2, 3, 1)),
        ht=np.full((2, 3), -100.0),
        coriolis_t=np.full((2, 3), 1e-4),
        dxt=np.array([2000.0, 3000.0]),
        dxu=np.array([2500.0, 3500.0]),
        dyt=np.array([2000.0, 3000.0, 4000.0]),
        dyu=np.array([2500.0, 3500.0, 4500.0]),
        area_t=np.full((2, 3), 6e6),
        area_u=np.full((2, 3), 7e6),
        area_v=np.full((2, 3), 8e6),
    )
    custom = veris.Experiment(grid=veris.Grid(2, 3), geometry=geometry)
    builtin = veris.Experiment.builtin(
        "island",
        grid=veris.Grid(4, 6),
        scenario={"islandWindSpeed": -3.0, "islandCooling": 70.0},
    )
    for definition in (custom, builtin):
        path = tmp_path / "resolved.json"
        resolved = definition.to_dict()
        path.write_text(json.dumps(resolved, allow_nan=False))
        restored = veris.Experiment.from_file(path)
        assert restored.to_dict() == resolved
        assert restored.configuration == definition.configuration
        assert restored.forcing.at(0).keys() == definition.forcing.at(0).keys()
        for name, value in definition.forcing.at(0).items():
            np.testing.assert_array_equal(restored.forcing.at(0)[name], value)
        if definition.geometry is not None:
            assert restored.geometry is not None
            for name in geometry.__dataclass_fields__:
                np.testing.assert_array_equal(
                    getattr(restored.geometry, name), getattr(geometry, name)
                )
    assert builtin.forcing.at(0)["Qnet"] == 70.0


def test_minimal_preset_file_uses_builtin_grid_and_physics_defaults(
    tmp_path: Path,
) -> None:
    """A declarative column needs no duplicate physics or grid declarations."""
    path = tmp_path / "column.json"
    path.write_text(
        json.dumps({"preset": "growth", "scenario": {"hIceMean": 0.7}, "steps": 2})
    )
    definition = veris.Experiment.from_file(path)
    assert definition.physics == "thermodynamics"
    assert definition.grid == veris.Grid(2, 2)
    assert definition.scenario["hIceMean"] == 0.7


def test_duration_conversion_is_independent_of_decimal_context(tmp_path: Path) -> None:
    """Other host libraries' Decimal precision cannot change model schedules."""
    path = tmp_path / "decimal.json"
    path.write_text(
        json.dumps(
            {"output": {"streams": [{"sampling_interval": {"seconds": 1.000001}}]}}
        )
    )
    with localcontext() as context:
        context.prec = 3
        definition = veris.Experiment.from_file(path)
    assert definition.output is not None
    assert definition.output.streams[0].sampling_interval == timedelta(
        microseconds=1000001
    )


def test_python_overrides_cannot_relabel_an_unsupported_file_schema(
    tmp_path: Path,
) -> None:
    """A schema version identifies the format of the file before any overrides."""
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="version"):
        veris.Experiment.from_file(path, overrides={"schema_version": 1})
