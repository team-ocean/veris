"""Preset commands retain numerical and file contracts through Simulation.

Frozen original trajectories anchor the scientific behavior. The
lifecycle spy wraps the real initializer and never replaces physics.
"""

import json
import os
import subprocess
import sys
from dataclasses import fields
from importlib import import_module
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import numpy as np
import pytest

import veris
from veris import Experiment, Grid, initialize_experiment
from veris.execution.advance import advance
from veris.io import read_record
from veris.setups import run_parallel


@pytest.mark.parametrize(
    "driver",
    ["run_dyn", "run_growth"],
)
def test_serial_cli_uses_real_lifecycle_and_preserves_reference_snapshots(
    tmp_path: Path,
    driver: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    steps = 2
    module = import_module(f"veris.setups.{driver}")
    recorded = []
    original = veris.Simulation.initialize

    def initialize(experiment: Any, **kwargs: Any) -> Any:
        recorded.append((experiment, kwargs))
        return original(experiment, **kwargs)

    monkeypatch.setattr(veris.Simulation, "initialize", staticmethod(initialize))
    history, final = tmp_path / "history.nc", tmp_path / "final.nc"
    extra = (
        [] if driver == "run_growth" else ["--nx", "6", "--ny", "8", "--evp-steps", "2"]
    )
    returned = module.main(
        [
            "--steps",
            str(steps),
            "--backend",
            "cpu",
            "--output-history",
            str(history),
            "--output-snapshot",
            str(final),
            "--calendar",
            "360_day",
            "--start-date",
            "2000-02-30",
            *extra,
        ]
    )
    assert returned is None and len(recorded) == 1
    definition, options = recorded[0]
    assert options["output_path"] == history
    assert definition.preset == ("growth" if driver == "run_growth" else "dynamics")
    if driver == "run_growth":
        expected, conf, phys = initialize_experiment(Experiment.builtin("growth"))
    else:
        expected, conf, phys = initialize_experiment(
            Experiment.builtin("dynamics", grid=Grid(6, 8), settings={"nEVPsteps": 2})
        )
    compiled_advance = jax.jit(
        advance, static_argnames=("configuration", "constants", "physics")
    )
    physics = "thermodynamics" if driver == "run_growth" else "dynamics"
    for _ in range(steps):
        expected = compiled_advance(expected, conf, phys, physics=physics)[0]
    saved = read_record(final)
    assert set(saved.fields) == {item.name for item in fields(expected)}
    assert saved.calendar == "360_day"
    assert saved.units == "seconds since 2000-02-30 00:00:00"
    duration = conf.deltatTherm if driver == "run_growth" else conf.deltatDyn
    assert saved.time == steps * duration
    for name, actual in saved.fields.items():
        reference = np.asarray(getattr(expected, name))[2:-2, 2:-2]
        error = np.abs(actual - reference)
        assert np.all(error <= 1e-12 * (1 + np.abs(reference))), (
            f"ERROR {driver} {name}: max_abs={error.max():.6g}"
        )
    sampled = read_record(history, stream="instantaneous")
    assert sampled.time == saved.time
    with h5netcdf.File(final, "r") as file:
        assert json.loads(file.attrs["experiment"])["preset"] == definition.preset
    output = capsys.readouterr().out
    assert (
        f"Saved {steps} growth steps"
        if driver == "run_growth"
        else f"dynamics: {steps} steps, cpu"
    ) in output


def test_final_only_cli_preserves_calendar_and_selected_fields(tmp_path: Path) -> None:
    from veris.setups import run_growth

    final = tmp_path / "selected.nc"
    run_growth.main(
        [
            "--steps",
            "0",
            "--output-snapshot",
            str(final),
            "--io-variables",
            "hIceMean,Area",
            "--calendar",
            "noleap",
            "--start-date",
            "2001-03-01",
        ]
    )
    saved = read_record(final)
    assert set(saved.fields) == {"hIceMean", "Area"}
    assert (
        saved.calendar == "noleap"
        and saved.units == "seconds since 2001-03-01 00:00:00"
    )


def test_real_local_parallel_cli_retains_nine_field_output_and_resolved_definition(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", PYTHONPATH=str(root)
    )
    for name in (
        "JAX_COORDINATOR_ADDRESS",
        "JAX_NUM_PROCESSES",
        "JAX_PROCESS_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
    ):
        environment.pop(name, None)
    final = tmp_path / "parallel.nc"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "veris.setups.run_parallel",
            "--nx",
            "12",
            "--ny",
            "16",
            "--mesh",
            "2",
            "2",
            "--steps",
            "2",
            "--evp-steps",
            "2",
            "--output-snapshot",
            str(final),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR lifecycle parallel CLI: {result.stderr[-2000:]}"
    )
    assert "mesh=2x2 backend=cpu steps=2" in result.stdout
    saved = read_record(final)
    assert set(saved.fields) == set(run_parallel.OUTPUT_FIELDS) and saved.time == 1200
    serial, conf, phys = initialize_experiment(
        Experiment.builtin("dynamics", grid=Grid(12, 16), settings={"nEVPsteps": 2})
    )
    compiled_advance = jax.jit(
        advance, static_argnames=("configuration", "constants", "physics")
    )
    for _ in range(2):
        serial = compiled_advance(serial, conf, phys, physics="dynamics")[0]
    for name, actual in saved.fields.items():
        reference = np.asarray(getattr(serial, name))[2:-2, 2:-2]
        assert actual.shape == (12, 16)
        error = np.abs(actual - reference)
        assert np.all(error <= 1e-10 * (1 + np.abs(reference))), (
            f"ERROR parallel {name}: max_abs={error.max():.6g}"
        )
    with h5netcdf.File(final, "r") as file:
        metadata = json.loads(file.attrs["experiment"])
    assert metadata["grid"] == {"nx": 12, "ny": 16}
    assert metadata["execution"]["partitions"] == [2, 2]


@pytest.mark.parametrize("driver", ["run_dyn", "run_growth"])
def test_module_driver_preserves_shared_runner_validation_error(
    driver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Module entry points propagate the canonical runner failure unchanged."""
    module = import_module(f"veris.setups.{driver}")
    error = ValueError("nonfinite output in hIceMean")

    def reject(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(module, "run_driver", reject)
    with pytest.raises(ValueError, match="nonfinite output") as failure:
        module.main(["--steps", "0", "--output-snapshot", str(tmp_path / "invalid.nc")])
    assert failure.value is error
    assert not (tmp_path / "invalid.nc").exists()
