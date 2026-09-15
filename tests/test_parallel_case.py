"""Parallel driver contracts, physical output layout, and launch validation."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from veris.io import read_record
from veris.setups import run_parallel


def test_four_cpu_dynamics_and_output_match_serial() -> None:
    """Use a fresh backend to check every local halo is removed before gather."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", PYTHONPATH=str(root)
    )
    result = subprocess.run(
        [sys.executable, str(root / "tests/parallel_case_probe.py")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR parallel dynamics probe: {result.stderr[-2500:]}"
    )
    assert "four CPU devices match serial output" in result.stdout


def test_explicit_bootstrap_and_incomplete_environment() -> None:
    """Explicit ranks bypass automatic cluster detection with CPU device zero."""
    environment = {
        "JAX_COORDINATOR_ADDRESS": "node001:12345",
        "JAX_NUM_PROCESSES": "4",
        "JAX_PROCESS_ID": "2",
    }
    assert run_parallel.distributed_options(environment, "cpu") == {
        "coordinator_address": "node001:12345",
        "num_processes": 4,
        "process_id": 2,
        "local_device_ids": [0],
    }
    with pytest.raises(ValueError, match="together"):
        run_parallel.distributed_options({"JAX_PROCESS_ID": "1"}, "cpu")
    environment["JAX_PROCESS_ID"] = "4"
    with pytest.raises(ValueError, match="process"):
        run_parallel.distributed_options(environment, "cpu")


def test_local_and_slurm_bootstrap() -> None:
    """An allocation alone must not start distributed rendezvous."""
    assert run_parallel.distributed_options({}, "cpu") is None
    assert run_parallel.distributed_options({"SLURM_JOB_ID": "42"}, "cpu") is None
    assert run_parallel.distributed_options(
        {"SLURM_NTASKS": "2", "SLURM_PROCID": "1"}, "cpu"
    ) == {"cluster_detection_method": "slurm", "local_device_ids": [0]}


def test_timed_loop_discards_warmup_and_evolves_state() -> None:
    """A pure recurrence detects accidental warmup advancement under scan.

    Python side effects cannot count traced transitions; the nonlinear final
    value distinguishes the original three-step trajectory from warmed state.
    """

    def increment(state: jax.Array) -> jax.Array:
        return 2 * state + 1

    state, warmup_seconds, elapsed_seconds = run_parallel.run_timed(
        jax.numpy.asarray(0), increment, 3
    )
    assert int(state) == 7
    assert warmup_seconds >= 0
    assert elapsed_seconds >= 0


def test_remove_halos_and_save_selected_fields(tmp_path: Path) -> None:
    """Output stores physical cells and only the nine reference fields."""
    mesh = jax.make_mesh((1, 1), ("x", "y"))
    expected = np.arange(30, dtype=float).reshape(5, 6)
    packed = np.pad(expected, 2, constant_values=-999)
    array = jax.device_put(
        packed, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y"))
    )
    interior = run_parallel.remove_halos(array, mesh)
    np.testing.assert_array_equal(interior, expected)
    fields = dict.fromkeys(run_parallel.OUTPUT_FIELDS, np.asarray(interior))
    destination = tmp_path / "subdir" / "output.nc"
    run_parallel.save_output(destination, fields, process_index=1)
    assert not destination.exists()
    run_parallel.save_output(destination, fields, process_index=0)
    result = read_record(destination)
    assert set(result.fields) == set(run_parallel.OUTPUT_FIELDS)
    np.testing.assert_array_equal(result.fields["hIceMean"], expected)


def test_invalid_mesh_and_cli_dimensions() -> None:
    """Bad extents fail before allocating model arrays."""
    with pytest.raises(ValueError, match="devices"):
        run_parallel.create_mesh((jax.device_count() + 1, 1), "cpu")
    for arguments in (
        ["--steps", "0"],
        ["--evp-steps", "0"],
        ["--mesh", "0", "1"],
        ["--nx", "-1"],
    ):
        with pytest.raises(SystemExit):
            run_parallel.parse_args(arguments)
    arguments = run_parallel.parse_args([])
    assert arguments.steps == 1000
    assert arguments.evp_steps == 120
    assert arguments.output == Path("parallel.nc")


def test_nonfinite_output_is_rejected_before_writing(tmp_path: Path) -> None:
    """A failed trajectory cannot leave a result file that appears successful."""
    output = dict.fromkeys(run_parallel.OUTPUT_FIELDS, np.zeros((2, 3)))
    output["hIceMean"] = np.full((2, 3), np.nan)
    destination = tmp_path / "invalid.nc"
    with pytest.raises(ValueError, match="nonfinite.*hIceMean"):
        run_parallel.save_output(destination, output, process_index=0)
    assert not destination.exists()


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="requires a GPU backend")
def test_gpu_cli_writes_finite_output(tmp_path: Path) -> None:
    """The public GPU CLI must initialize the installed accelerator backend."""
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "gpu.nc"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "veris.setups.run_parallel",
            "--backend",
            "gpu",
            "--nx",
            "12",
            "--ny",
            "16",
            "--steps",
            "2",
            "--evp-steps",
            "4",
            "--output",
            str(destination),
        ],
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, f"ERROR parallel GPU CLI: {result.stderr[-2500:]}"
    fields = read_record(destination).fields
    assert set(fields) == set(run_parallel.OUTPUT_FIELDS)
    assert all(value.shape == (12, 16) for value in fields.values())
    assert all(np.isfinite(value).all() for value in fields.values())


def test_parallel_cli_rejects_nonfinite_trajectory_before_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final driver output validates collected fields before creating netCDF."""
    from dataclasses import replace

    from veris._typing import State

    def nonfinite_trajectory(
        state: State, *args: Any, **kwargs: Any
    ) -> tuple[State, float, float]:
        """Represent a numerical failure after the timed integration."""
        return replace(state, hIceMean=state.hIceMean * np.nan), 0.0, 0.0

    monkeypatch.setattr(run_parallel, "run_timed", nonfinite_trajectory)
    path = tmp_path / "failed-trajectory.nc"
    with pytest.raises(ValueError, match="nonfinite.*hIceMean"):
        run_parallel.main(
            [
                "--nx",
                "4",
                "--ny",
                "4",
                "--steps",
                "1",
                "--evp-steps",
                "2",
                "--output",
                str(path),
            ]
        )
    assert not path.exists()
