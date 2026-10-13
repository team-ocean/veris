"""Real two-process scientific/output parity for every mode and service ownership."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import h5netcdf
import numpy as np
import pytest

from veris import run_experiment
from veris.io import read_record


def _launch(
    root: Path,
    kind: str,
    cyclic_y: bool,
    extra: tuple[str, ...] = (),
    *,
    closed_ad: bool = False,
) -> None:
    project = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = []
    try:
        for rank in range(2):
            environment = dict(
                os.environ,
                PYTHONPATH=str(project),
                JAX_PLATFORMS="cpu",
                JAX_NUM_CPU_DEVICES="1",
                JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}",
                JAX_NUM_PROCESSES="2",
                JAX_PROCESS_ID=str(rank),
            )
            for name in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"):
                environment.pop(name, None)
            command = [
                sys.executable,
                str(project / "tests/distributed_framework_probe.py"),
                "--kind",
                kind,
                "--cyclic-y",
                str(int(cyclic_y)),
                "--output-root",
                str(root),
                *extra,
            ]
            if closed_ad:
                command = [
                    sys.executable,
                    str(project / "tests/closed_y_sharding_probe.py"),
                    "--distributed",
                ]
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for rank, process in enumerate(processes):
            _, stderr = process.communicate(timeout=360 if closed_ad else 150)
            assert process.returncode == 0, (
                f"ERROR distributed rank{rank}: {stderr[-2200:]}"
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["island", "dynamics", "growth", "custom"])
@pytest.mark.parametrize("cyclic_y", [False, True])
def test_two_process_all_modes_spatial_forcing_history_and_clock(
    tmp_path: Path, kind: str, cyclic_y: bool
) -> None:
    from tests.distributed_framework_probe import definition

    distributed, serial = tmp_path / "distributed", tmp_path / "serial"
    distributed.mkdir()
    serial.mkdir()
    _launch(distributed, kind, cyclic_y)
    expected_result = run_experiment(
        definition(kind, cyclic_y),
        output_path=serial / "history.nc",
        snapshot_path=serial / "final.nc",
    )
    expected, actual = (
        read_record(serial / "final.nc"),
        read_record(distributed / "final.nc"),
    )
    assert len(actual.fields) == 70 and set(actual.fields) == set(expected.fields)
    assert (
        actual.time
        == expected.time
        == expected_result.elapsed.total_seconds()
        == 1200.0
    )
    for name, reference in expected.fields.items():
        error = np.abs(actual.fields[name] - reference)
        assert np.all(error <= 1e-10 * (1 + np.abs(reference))), (
            f"ERROR distributed {kind} {name}: max_abs={error.max():.6g}"
        )
    with (
        h5netcdf.File(serial / "history.nc", "r") as expected_history,
        h5netcdf.File(distributed / "history.nc", "r") as actual_history,
    ):
        for stream in ("instant", "mean"):
            expected_group = expected_history.groups[stream]
            actual_group = actual_history.groups[stream]
            assert set(actual_group.variables) == set(expected_group.variables)
            for name, reference in expected_group.variables.items():
                np.testing.assert_allclose(
                    actual_group.variables[name][:],
                    reference[:],
                    rtol=1e-10,
                    atol=1e-10,
                )
    for rank in range(2):
        status = json.loads((distributed / f"rank-{rank}.json").read_text())
        assert (
            status["rank"] == rank
            and status["closed_owned"]
            and status["error"] is None
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "extra",
    [
        ("--borrowed",),
        ("--borrowed", "--local-cpu-devices", "2"),
        ("--snapshot-failure",),
    ],
)
def test_real_distributed_borrowing_and_snapshot_exception_cleanup(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    _launch(tmp_path, "dynamics", True, extra)
    for rank in range(2):
        status = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        if "--borrowed" in extra:
            assert (
                status["borrowed"]
                and not status["closed_owned"]
                and status["error"] is None
            )
        else:
            assert (
                status["closed_owned"]
                and status["error"] == "injected snapshot failure"
            )
    if extra == ("--snapshot-failure",):
        assert not (tmp_path / "final.nc").exists()


@pytest.mark.slow
@pytest.mark.parametrize("partitions", [(1, 2), (2, 1)])
def test_repeated_nested_evp_rollouts_match_all_serial_fields(
    tmp_path: Path, partitions: tuple[int, int]
) -> None:
    """Exercise the intermittent nested-loop/final-halo communication failure."""
    from tests.distributed_framework_probe import repeated_dynamics_definition

    run_experiment(
        repeated_dynamics_definition(), snapshot_path=tmp_path / "reference.nc"
    )
    _launch(
        tmp_path,
        "dynamics",
        True,
        ("--repeat-dynamics", "--dynamics-mesh", *(str(p) for p in partitions)),
    )


@pytest.mark.slow
def test_distributed_closed_coupled_derivatives_after_halo_prelude(
    tmp_path: Path,
) -> None:
    """Preserve mixed halo/AD history before both closed-wall slip choices."""
    _launch(tmp_path, "island", False, closed_ad=True)
