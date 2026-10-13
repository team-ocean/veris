"""Real public lifecycle communication and output checks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_public_lifecycle_four_cpu_layouts_presets_and_custom_geometry() -> None:
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
    result = subprocess.run(
        [sys.executable, str(root / "tests/lifecycle_sharding_probe.py")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR public lifecycle sharding: {result.stderr[-2500:]}"
    )
    assert (
        "24 four-CPU preset/custom initialization, run and output cases passed"
        in result.stdout
    )
