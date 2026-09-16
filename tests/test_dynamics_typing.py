"""Concrete model states remain valid compiled-kernel inputs."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("case", ["immutable", "wrong-strength", "wrong-stress"])
def test_concrete_dynamics_contract(tmp_path: Path, case: str) -> None:
    """Check both public signatures in one valid module; reject each wrong input."""
    valid = case == "immutable"
    source = f"""from jax import Array
from veris._typing import State
from numpy import float64
from numpy.typing import NDArray
from veris.dynamics_routines import SeaIceStrength, stress
from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants

def strength(state: {"str" if case == "wrong-strength" else "State"}, settings: Configuration, phys: PhysicalConstants) -> Array:
    return SeaIceStrength(state, settings, phys)

def stresses(state: {"str" if case == "wrong-stress" else "State"}, settings: Configuration, phys: PhysicalConstants, field: NDArray[float64]) -> tuple[Array, Array, Array]:
    return stress(state, settings, phys, field, field, field, field, field, field)
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
