"""Fresh-process device topologies validate scheduled sharded reductions."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_probe(backend: str) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cuda,cpu" if backend == "gpu" else "cpu",
        PYTHONPATH=str(root),
    )
    if backend == "cpu":
        environment["JAX_NUM_CPU_DEVICES"] = "4"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tests/output_scan_probe.py"),
            "--backend",
            backend,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR scheduled {backend} output probe: {result.stderr[-2500:]}"
    )
    assert f"scheduled output passed on {backend}" in result.stdout


def test_scheduled_output_four_cpu_devices() -> None:
    """Real four-device sums remain sharded until record-only collection."""
    _run_probe("cpu")


@pytest.mark.skipif(
    not {"gpu", "cuda"}.intersection(os.environ.get("JAX_PLATFORMS", "cpu").split(",")),
    reason="requires explicitly requested GPU backend and two devices",
)
def test_scheduled_output_two_gpu_devices() -> None:
    """A requested GPU run must use two actual GPUs and cannot fall back to CPU."""
    _run_probe("gpu")
