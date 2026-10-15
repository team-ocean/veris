"""Every maintained CLI uses the same runner and authored grouped overrides."""

import importlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from click.testing import CliRunner

import veris
from veris.io import read_record
from veris.model.types import State


def _file(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps({"schema_version": 1, **data}))
    return path


@pytest.mark.parametrize("cyclic_y", [False, True])
def test_central_cli_grouped_overrides_match_python_all_fields(
    cyclic_y: bool, tmp_path: Path
) -> None:
    from veris.cli import cli
    from veris.model.experiment_file import experiment_from_dict

    kind = "island"
    raw = {
        "preset": kind,
        "grid": {"nx": 4, "ny": 6},
        "steps": 3,
        "settings": {"nEVPsteps": 2, "nITC": 1, "deltatDyn": 60.0, "deltatTherm": 60.0},
    }
    source = _file(tmp_path / "experiment.json", raw)
    overrides = {
        "steps": 1,
        "settings": {"enable_cyclic_y": cyclic_y},
        "forcing": {"constant": {"uWind": 0.0}},
        "initial_conditions": {"hIceMean": 0.4},
    }
    expected = veris.run_experiment(experiment_from_dict(raw), overrides=overrides)
    output = tmp_path / "final.nc"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(source),
            "--steps",
            "1",
            "--output-snapshot",
            str(output),
            "--set",
            f"settings.enable_cyclic_y={json.dumps(cyclic_y)}",
            "--set",
            "forcing.constant.uWind=0.0",
            "--set",
            "initial_conditions.hIceMean=0.4",
        ],
    )
    assert result.exit_code == 0, f"ERROR grouped CLI {kind}: {result.output[-1500:]}"
    saved = read_record(output)
    assert saved.time == expected.elapsed.total_seconds()
    for item in fields(State):
        actual = saved.fields[item.name]
        reference = np.asarray(getattr(expected.state, item.name))[2:-2, 2:-2]
        error = np.abs(actual - reference)
        assert np.all(error <= 1e-12 * (1 + np.abs(reference))), (
            f"ERROR {kind} CLI {item.name}: max_abs={error.max():.6g}"
        )


@pytest.mark.parametrize(
    "override",
    [
        "settings.typo=1",
        "settings..dtype=1",
        "settings.dtype=not_json",
        'output.streams.0.name="foo"',
        "geometry.maskT=NaN",
        "settings.dtype",
    ],
)
def test_bad_grouped_override_is_a_concise_error_without_effects(
    override: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from veris.cli import cli

    source = _file(tmp_path / "experiment.json", {"preset": "growth", "steps": 0})
    output = tmp_path / "new" / "final.nc"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR invalid CLI override selected devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    result = CliRunner().invoke(
        cli, ["run", str(source), "--output-snapshot", str(output), "--set", override]
    )
    assert result.exit_code != 0 and "Error:" in result.output
    assert "Traceback" not in result.output and not output.parent.exists()


@pytest.mark.parametrize(
    "dedicated,generic",
    [
        (["--steps", "0"], "steps=1"),
        (["--float-type", "float64"], 'settings.dtype="float32"'),
        (["--backend", "cpu"], 'execution.backend="gpu"'),
        (["--mesh", "1", "2"], "execution.partitions=[2,1]"),
    ],
)
def test_dedicated_and_generic_alias_conflicts_are_rejected(
    dedicated: list[str], generic: str, tmp_path: Path
) -> None:
    from veris.cli import cli

    source = _file(tmp_path / "experiment.json", {"preset": "growth", "steps": 0})
    output = tmp_path / "new" / "final.nc"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(source),
            "--output-snapshot",
            str(output),
            *dedicated,
            "--set",
            generic,
        ],
    )
    assert result.exit_code != 0 and "conflict" in result.output.lower()
    assert not output.parent.exists()


@pytest.mark.parametrize("driver", ["central", "run_dyn", "run_growth", "run_parallel"])
def test_all_cli_entry_points_delegate_once_to_public_runner(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = veris.run_experiment

    def run(definition: veris.Experiment, **options: Any) -> veris.RunResult:
        calls.append((definition, options))
        return original(definition, **options)

    monkeypatch.setattr(veris, "run_experiment", run)
    output = tmp_path / "final.nc"
    if driver == "central":
        from veris.cli import cli

        source = _file(tmp_path / "experiment.json", {"preset": "growth", "steps": 0})
        result = CliRunner().invoke(
            cli, ["run", str(source), "--output-snapshot", str(output)]
        )
        assert result.exit_code == 0, result.output
    else:
        module = importlib.import_module(f"veris.setups.{driver}")
        extra = (
            []
            if driver == "run_growth"
            else ["--nx", "4", "--ny", "6", "--evp-steps", "2"]
        )
        module.main(["--steps", "0", "--output-snapshot", str(output), *extra])
    assert len(calls) == 1, f"ERROR {driver} bypasses public runner"
    assert output.exists()
    assert calls[0][1]["snapshot_path"] == output


def test_module_generic_controls_override_only_authored_flags(tmp_path: Path) -> None:
    from veris.setups import run_dyn

    output = tmp_path / "final.nc"
    run_dyn.main(
        [
            "--steps",
            "1",
            "--nx",
            "4",
            "--ny",
            "6",
            "--output-snapshot",
            str(output),
            "--set",
            "settings.nEVPsteps=2",
            "--set",
            "scenario.wind_speed=0.0",
            "--set",
            'settings.dtype="float32"',
        ]
    )
    saved = read_record(output)
    assert saved.fields["hIceMean"].dtype == np.dtype("float32")
    np.testing.assert_array_equal(saved.fields["uWind"], 0.0)


def test_cli_dtype_promotion_re_resolves_unrounded_preset_defaults(
    tmp_path: Path,
) -> None:
    from veris.cli import cli

    source = _file(
        tmp_path / "experiment.json",
        {"preset": "growth", "steps": 0, "settings": {"dtype": "float32"}},
    )
    output = tmp_path / "final.nc"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(source),
            "--float-type",
            "float64",
            "--output-snapshot",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    saved = read_record(output)
    np.testing.assert_array_equal(saved.fields["hIceMean"], 1.3)


def test_custom_geometry_cli_matches_python_all_physical_fields(tmp_path: Path) -> None:
    cyclic_y = False
    from tests.lifecycle_sharding_probe import _custom
    from veris.cli import cli
    from veris.io import OutputSettings

    definition = _custom(
        veris.Grid(8, 12), cyclic_y, OutputSettings(enabled=False)
    ).with_overrides({"steps": 1})
    source = _file(tmp_path / "geometry.json", definition.to_dict())
    expected = veris.run_experiment(definition)
    output = tmp_path / "final.nc"
    result = CliRunner().invoke(
        cli, ["run", str(source), "--output-snapshot", str(output)]
    )
    assert result.exit_code == 0, result.output
    actual = read_record(output)
    for item in fields(State):
        reference = np.asarray(getattr(expected.state, item.name))[2:-2, 2:-2]
        error = np.abs(actual.fields[item.name] - reference)
        assert np.all(error <= 1e-12 * (1 + np.abs(reference))), (
            f"ERROR custom CLI {item.name}: max_abs={error.max():.6g}"
        )


@pytest.mark.parametrize("kind", ["dynamics", "growth"])
def test_module_cli_nonzero_run_matches_python_boundary_policy(
    tmp_path: Path, kind: str
) -> None:
    cyclic_y = False
    driver = importlib.import_module(
        "veris.setups.run_dyn" if kind == "dynamics" else "veris.setups.run_growth"
    )
    controls = {"steps": 1, "settings": {"enable_cyclic_y": cyclic_y, "nEVPsteps": 2}}
    extra = []
    if kind == "dynamics":
        controls["grid"] = {"nx": 4, "ny": 6}
        extra = ["--nx", "4", "--ny", "6", "--evp-steps", "2"]
    expected = veris.run_experiment(
        veris.Experiment.builtin(kind, profile="standalone").with_overrides(controls)
    )
    output = tmp_path / "module.nc"
    driver.main(
        [
            "--steps",
            "1",
            "--output-snapshot",
            str(output),
            "--set",
            f"settings.enable_cyclic_y={json.dumps(cyclic_y)}",
            *extra,
        ]
    )
    actual = read_record(output)
    assert actual.time == expected.elapsed.total_seconds()
    for item in fields(State):
        reference = np.asarray(getattr(expected.state, item.name))[2:-2, 2:-2]
        error = np.abs(actual.fields[item.name] - reference)
        assert np.all(error <= 1e-12 * (1 + np.abs(reference))), (
            f"ERROR module {kind} {item.name}: max_abs={error.max():.6g}"
        )


@pytest.mark.parametrize("module", ["run_dyn", "run_growth", "run_parallel"])
def test_module_console_reports_bad_grouped_controls_without_traceback_or_files(
    tmp_path: Path, module: str
) -> None:
    import os
    import subprocess
    import sys

    output = tmp_path / "new" / "final.nc"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"veris.setups.{module}",
            "--steps",
            "0",
            "--output-snapshot",
            str(output),
            "--set",
            "settings.typo=1",
        ],
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Error:" in result.stderr and "typo" in result.stderr
    assert "Traceback" not in result.stderr and not output.parent.exists()
