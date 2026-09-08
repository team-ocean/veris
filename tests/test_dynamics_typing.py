"""Minimal immutable constitutive states remain valid compiled-kernel inputs."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("valid", [True, False], ids=["immutable", "wrong-mask"])
@pytest.mark.parametrize("kernel", ["strength", "stress"])
def test_minimal_dynamics_contract(tmp_path: Path, valid: bool, kernel: str) -> None:
    """Reject invalid masks without requiring unrelated fields or writable state."""
    fields = "    hIceMean: Array\n    Area: Array\n" if kernel == "strength" else ""
    settings = (
        "    pStar: float\n    cStar: float"
        if kernel == "strength"
        else "    noSlip: bool"
    )
    expression = (
        "SeaIceStrength(state, constants)"
        if kernel == "strength"
        else "stress(state, constants, field, field, field, field, field, field)"
    )
    returns = "Array" if kernel == "strength" else "tuple[Array, Array, Array]"
    source = f"""from typing import NamedTuple
from jax import Array
from numpy import float64
from numpy.typing import NDArray
from veris.dynamics_routines import SeaIceStrength, stress

class State(NamedTuple):
{fields}    iceMask: {"Array" if valid else "str"}

class Constants(NamedTuple):
{settings}

def evaluate(state: State, constants: Constants, field: NDArray[float64]) -> {returns}:
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
        assert result.returncode != 0, "ERROR non-array dynamics mask accepted"
        assert "invalid-argument-type" in diagnostic, diagnostic[-2000:]
