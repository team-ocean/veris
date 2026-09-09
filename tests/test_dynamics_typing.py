"""Concrete model states remain valid compiled-kernel inputs."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("valid", [True, False], ids=["immutable", "wrong-state"])
@pytest.mark.parametrize("kernel", ["strength", "stress"])
def test_concrete_dynamics_contract(tmp_path: Path, valid: bool, kernel: str) -> None:
    """Accept the concrete State and reject invalid state arguments."""
    expression = (
        "SeaIceStrength(state, settings, phys)"
        if kernel == "strength"
        else "stress(state, settings, phys, field, field, field, field, field, field)"
    )
    returns = "Array" if kernel == "strength" else "tuple[Array, Array, Array]"
    source = f"""from jax import Array
from veris.state import State
from numpy import float64
from numpy.typing import NDArray
from veris.dynamics_routines import SeaIceStrength, stress
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants

def evaluate(state: {"State" if valid else "str"}, settings: Settings, phys: PhysicalConstants, field: NDArray[float64]) -> {returns}:
    return {expression}
"""
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "dynamics_contract.py"
    path.write_text(source)
    result = subprocess.run(
        ["ty", "check", "--extra-search-path", str(root), str(path)],
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    diagnostic = result.stdout + result.stderr
    if valid:
        assert result.returncode == 0, diagnostic[-2000:]
    else:
        assert result.returncode != 0, "ERROR invalid dynamics state accepted"
        assert "invalid-argument-type" in diagnostic, diagnostic[-2000:]
