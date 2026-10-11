"""Fresh processes verify serial resource ownership under real ambient meshes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("devices", [2, 4])
def test_serial_lifecycle_suspends_and_restores_ambient_mesh(devices: int) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_NUM_CPU_DEVICES=str(devices),
        PYTHONPATH=str(root),
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
        [sys.executable, str(root / "tests/serial_mesh_context_probe.py")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, f"ERROR serial ambient mesh: {result.stderr[-2500:]}"
    assert (
        f"serial ambient mesh: {devices} CPU devices, four lifecycle contexts passed"
        in result.stdout
    )
