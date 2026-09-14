"""Verify zero-strain AD across real device partitions in a fresh backend."""

import os
import subprocess
import sys
from pathlib import Path


def test_zero_strain_ad_on_four_cpu_devices() -> None:
    """Partition boundaries preserve the corrected evolving sensitivities."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", PYTHONPATH=str(root)
    )
    result = subprocess.run(
        [sys.executable, str(root / "tests/ad_sharding_probe.py")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, f"ERROR sharded AD probe: {result.stderr[-2500:]}"
    assert "evolving JVP/VJP/FD passed" in result.stdout
