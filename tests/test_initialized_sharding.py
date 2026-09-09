"""Public registry initialization supports a complete four-device coupled run."""

import os
import subprocess
import sys
from pathlib import Path


def test_initialized_four_cpu_state_step_and_forcing_gradients() -> None:
    """Fresh-process CPU topology verifies all State leaves and coupled AD."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", PYTHONPATH=str(root)
    )
    result = subprocess.run(
        [sys.executable, str(root / "tests/initialized_sharding_probe.py")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR initialized sharding probe: {result.stderr[-2500:]}"
    )
    assert "coupled step, JVP and VJP passed on four CPU devices" in result.stdout
