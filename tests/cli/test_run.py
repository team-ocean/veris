"""CLI file execution reuses real model validation, precision and lifecycle I/O."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5netcdf
import jax
import numpy as np
import pytest
from click.testing import CliRunner

from veris import Execution, Experiment, Grid, Simulation
from veris.io import read_record


def write_definition(path: Path, **options: object) -> Path:
    """Write independent compact JSON without reusing the loader under test."""
    path.write_text(
        json.dumps({"schema_version": 1, "preset": "growth", "steps": 2, **options})
    )
    return path


def test_python_factory_can_import_sibling_and_keeps_callers_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "cli_test_helper.py").write_text("STEPS = 3\n")
    setup = tmp_path / "setup.py"
    setup.write_text(
        "from veris import Experiment\nfrom cli_test_helper import STEPS\ndef build_experiment():\n    return Experiment.builtin('growth', steps=STEPS)\n"
    )
    before = list(sys.path)
    loader = importlib.import_module("veris.cli.run")
    definition = loader.load_experiment(setup)
    assert definition.preset == "growth" and definition.steps == 3
    assert sys.path == before


@pytest.mark.parametrize(
    "source,reason",
    [
        ("# no factory\n", "build_experiment"),
        ("build_experiment = 1\n", "build_experiment"),
        ("def build_experiment():\n    return None\n", "Experiment"),
        (
            "def build_experiment():\n    raise ValueError('bad forcing')\n",
            "bad forcing",
        ),
        ("invalid python !!!\n", "syntax"),
    ],
)
def test_invalid_python_setup_is_a_concise_cli_error(
    source: str, reason: str, tmp_path: Path
) -> None:
    setup = tmp_path / "bad.py"
    setup.write_text(source)
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(
        cli, ["run", str(setup), "--output", str(tmp_path / "final.nc")]
    )
    assert result.exit_code != 0
    assert "Error:" in result.output and reason.lower() in result.output.lower()
    assert "Traceback" not in result.output
    assert not (tmp_path / "final.nc").exists()


def test_json_preserves_file_values_when_flags_are_omitted(tmp_path: Path) -> None:
    source = write_definition(
        tmp_path / "experiment.json",
        settings={"dtype": "float32"},
        execution={"backend": "cpu"},
    )
    loader = importlib.import_module("veris.cli.run")
    definition = loader.load_experiment(source)
    assert definition.steps == 2 and definition.configuration.dtype == "float32"
    assert definition.execution == Execution(backend="cpu")
    final = tmp_path / "preserved.nc"
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(cli, ["run", str(source), "--output", str(final)])
    assert result.exit_code == 0, result.output
    saved = read_record(final)
    assert saved.time == 172800 and saved.fields["Area"].dtype == np.dtype("float32")
    with h5netcdf.File(final, "r") as file:
        metadata = json.loads(file.attrs["experiment"])
    assert metadata["steps"] == 2 and metadata["execution"]["backend"] == "cpu"


@pytest.mark.parametrize("steps,dtype", [(0, "float64"), (2, "float32")])
def test_json_run_matches_direct_lifecycle_and_records_overrides(
    tmp_path: Path, steps: int, dtype: str
) -> None:
    # Resolved float64 JSON includes constants.dtype: changing only settings
    # would reject the float32 override before any simulation can run.
    source = tmp_path / "experiment.json"
    source.write_text(json.dumps(Experiment.builtin("growth", steps=3).to_dict()))
    final = tmp_path / "nested" / "final.nc"
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(source),
            "--steps",
            str(steps),
            "--float-type",
            dtype,
            "--backend",
            "cpu",
            "--output",
            str(final),
            "--io-variables",
            "hIceMean,Area",
        ],
    )
    assert result.exit_code == 0, result.output
    record = read_record(final)
    assert set(record.fields) == {"hIceMean", "Area"}
    assert record.time == steps * 86400
    assert record.fields["hIceMean"].dtype == np.dtype(dtype)
    with Simulation.initialize(
        Experiment.builtin(
            "growth",
            steps=steps,
            settings={"dtype": dtype},
            execution=Execution(backend="cpu"),
        )
    ) as simulation:
        state = simulation.run().state
    for name, value in record.fields.items():
        expected = np.asarray(getattr(state, name))[2:-2, 2:-2]
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("preset", ["growth", "dynamics", "island"])
def test_copied_builtin_factory_runs_with_its_preset(
    preset: str, tmp_path: Path
) -> None:
    registry = importlib.import_module("veris.cli.setups")
    directory = registry.copy_setup(preset, tmp_path / "copy")
    loader = importlib.import_module("veris.cli.run")
    definition = loader.load_experiment(directory / "setup.py")
    assert definition.preset == preset and definition.steps > 0
    output = tmp_path / "output.nc"
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(directory / "setup.py"),
            "--steps",
            "1" if preset == "island" else "0",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    saved = read_record(output)
    assert saved.fields["hIceMean"].shape == (definition.grid.nx, definition.grid.ny)
    assert saved.time == (
        definition.step_duration.total_seconds() if preset == "island" else 0
    )


@pytest.mark.parametrize(
    "extra,reason",
    [
        (["--io-variables", "typo"], "variables"),
        (["--io-variables", "Area,Area"], "variables"),
        (["--steps", "-1"], "range"),
        (["--mesh", "3", "1"], "partitions"),
    ],
)
def test_invalid_run_does_not_create_output_directories(
    extra: list[str], reason: str, tmp_path: Path
) -> None:
    source = write_definition(tmp_path / "experiment.json")
    final = tmp_path / "missing" / "final.nc"
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(
        cli, ["run", str(source), "--output", str(final), *extra]
    )
    assert result.exit_code != 0 and reason in result.output
    assert not final.parent.exists()


def test_history_preserves_json_calendar_and_schedule(tmp_path: Path) -> None:
    source = write_definition(
        tmp_path / "experiment.json",
        output={
            "calendar": "360_day",
            "start": {"year": 2000, "month": 2, "day": 30},
            "streams": [
                {
                    "name": "daily",
                    "variables": ["Area"],
                    "sampling_interval": {"seconds": 86400},
                    "period": "daily",
                }
            ],
        },
    )
    final, history = tmp_path / "final.nc", tmp_path / "history.nc"
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(
        cli, ["run", str(source), "--output", str(final), "--netcdf", str(history)]
    )
    assert result.exit_code == 0, result.output
    record = read_record(history, stream="daily")
    assert record.calendar == "360_day" and set(record.fields) == {"Area"}
    assert record.units == "seconds since 2000-02-30 00:00:00"


def test_default_history_samples_selected_fields_each_step(tmp_path: Path) -> None:
    source = write_definition(tmp_path / "history.json", settings={"dtype": "float32"})
    cli = importlib.import_module("veris.cli").cli
    final, history = tmp_path / "final.nc", tmp_path / "nested" / "history.nc"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(source),
            "--output",
            str(final),
            "--netcdf",
            str(history),
            "--io-variables",
            "Area",
        ],
    )
    assert result.exit_code == 0, result.output
    with h5netcdf.File(history, "r") as file:
        np.testing.assert_array_equal(
            file.groups["instantaneous"].variables["time"][:], [0, 86400, 172800]
        )
    saved = read_record(history, stream="instantaneous")
    assert set(saved.fields) == {"Area"} and saved.fields["Area"].dtype == np.dtype(
        "float32"
    )


def test_enabled_output_requires_history_destination(tmp_path: Path) -> None:
    source = write_definition(tmp_path / "enabled.json", output={"enabled": True})
    final = tmp_path / "uncreated" / "final.nc"
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(cli, ["run", str(source), "--output", str(final)])
    assert result.exit_code != 0 and "--netcdf" in result.output
    assert not final.parent.exists()


def test_output_destinations_cannot_alias_or_overwrite(tmp_path: Path) -> None:
    source = write_definition(tmp_path / "experiment.json")
    final = tmp_path / "final.nc"
    cli = importlib.import_module("veris.cli").cli
    same = CliRunner().invoke(
        cli, ["run", str(source), "--output", str(final), "--netcdf", str(final)]
    )
    assert same.exit_code != 0 and "distinct" in same.output and not final.exists()
    final.write_bytes(b"keep user data")
    result = CliRunner().invoke(cli, ["run", str(source), "--output", str(final)])
    assert result.exit_code != 0 and final.read_bytes() == b"keep user data"


def test_real_cpu_mesh_cli_matches_serial(tmp_path: Path) -> None:
    source = write_definition(
        tmp_path / "mesh.json",
        preset="dynamics",
        grid={"nx": 6, "ny": 8},
        settings={"nEVPsteps": 2},
        steps=1,
    )
    final = tmp_path / "mesh.nc"
    environment = dict(os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="2")
    for name in (
        "JAX_COORDINATOR_ADDRESS",
        "JAX_NUM_PROCESSES",
        "JAX_PROCESS_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "veris.cli",
            "run",
            str(source),
            "--mesh",
            "1",
            "2",
            "--backend",
            "cpu",
            "--output",
            str(final),
            "--io-variables",
            "hIceMean,Area,uIce,vIce",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1800:]
    saved = read_record(final)
    with Simulation.initialize(
        Experiment.builtin(
            "dynamics",
            grid=Grid(6, 8),
            settings={"nEVPsteps": 2},
            steps=1,
            execution=Execution(backend="cpu"),
        )
    ) as simulation:
        state = simulation.run().state
    for name, value in saved.fields.items():
        expected = np.asarray(getattr(state, name))[2:-2, 2:-2]
        error = np.abs(value - expected)
        assert np.all(error <= 1e-10 * (1 + np.abs(expected))), (
            f"ERROR {name}: max_abs={error.max():.6g}"
        )


@pytest.mark.parametrize("mesh", [None, (1, 2)])
def test_gpu_central_cli_writes_float32_output(
    tmp_path: Path, mesh: tuple[int, int] | None
) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("requires native GPU runtime")
    source = write_definition(
        tmp_path / "gpu.json",
        **(
            {}
            if mesh is None
            else {
                "preset": "dynamics",
                "grid": {"nx": 6, "ny": 8},
                "settings": {"nEVPsteps": 2},
                "steps": 1,
            }
        ),
    )
    cli = importlib.import_module("veris.cli").cli
    final = tmp_path / "gpu.nc"
    result = CliRunner().invoke(
        cli,
        [
            "run",
            str(source),
            "--backend",
            "gpu",
            "--float-type",
            "float32",
            "--output",
            str(final),
            *([] if mesh is None else ["--mesh", *map(str, mesh)]),
        ],
    )
    assert result.exit_code == 0, result.output
    record = read_record(final)
    assert record.fields["hIceMean"].dtype == np.dtype("float32")
    assert all(np.isfinite(value).all() for value in record.fields.values())


def test_reference_trace_loglevel_is_accepted(tmp_path: Path) -> None:
    source = write_definition(tmp_path / "trace.json", steps=0)
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(
        cli, ["run", str(source), "-v", "trace", "--output", str(tmp_path / "trace.nc")]
    )
    assert result.exit_code == 0, result.output


def test_distributed_run_requires_backend_before_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_definition(
        tmp_path / "distributed.json",
        preset="dynamics",
        grid={"nx": 6, "ny": 8},
        execution={"partitions": [1, 2]},
    )
    monkeypatch.setenv("JAX_COORDINATOR_ADDRESS", "localhost:12345")
    monkeypatch.setenv("JAX_NUM_PROCESSES", "2")
    monkeypatch.setenv("JAX_PROCESS_ID", "0")

    def forbid_initialize(**kwargs: object) -> None:
        raise AssertionError(
            "ERROR ambiguous backend reached distributed initialization"
        )

    monkeypatch.setattr(jax.distributed, "initialize", forbid_initialize)
    cli = importlib.import_module("veris.cli").cli
    final = tmp_path / "uncreated" / "final.nc"
    result = CliRunner().invoke(cli, ["run", str(source), "--output", str(final)])
    assert (
        result.exit_code != 0
        and "explicit" in result.output
        and "backend" in result.output
    )
    assert not final.parent.exists()
