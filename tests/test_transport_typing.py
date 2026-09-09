"""Directional flux contracts accept concrete inputs through compilation."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("valid", [True, False], ids=["immutable", "wrong-state"])
@pytest.mark.parametrize("direction", ["Zonal", "Meridional"])
def test_directional_transport_contract(
    tmp_path: Path, valid: bool, direction: str
) -> None:
    """Accept NumPy fields and concrete State, but reject invalid state arguments."""
    source = f"""from veris.state import State
from veris.configuration import Settings
from jax import Array
from numpy import float64
from numpy.typing import NDArray
from veris.advection import calc_{direction}Flux
from veris.physical_constants import PhysicalConstants

def evaluate(state: {"State" if valid else "str"}, constants: Settings, field: NDArray[float64]) -> Array:
    return calc_{direction}Flux(state, constants, PhysicalConstants(), field, field)
"""
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "transport_contract.py"
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
        assert result.returncode != 0, "ERROR invalid transport state accepted"
        assert "invalid-argument-type" in diagnostic, diagnostic[-2000:]


def test_boolean_ocean_masks_remain_valid_inputs(tmp_path: Path) -> None:
    """Typing must retain existing boolean-mask support in bulk and halo APIs."""
    root = Path(__file__).resolve().parents[1]
    source = """from jax import Array
from numpy import bool_, float64
from numpy.typing import NDArray
from veris.fill_overlap import fill_overlap
from veris.heat_flux_CESM import dqnetdt
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants

def evaluate(sett: Settings, phys: PhysicalConstants, mask: NDArray[bool_], field: NDArray[float64]) -> tuple[Array, Array, Array]:
    fill_overlap(mask, sett)
    return dqnetdt(sett, phys, mask, field, field, field, field, field, field, field)
"""
    path = tmp_path / "boolean_mask.py"
    path.write_text(source)
    result = subprocess.run(
        ["ty", "check", "--extra-search-path", str(root), str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
