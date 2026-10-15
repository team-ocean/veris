"""Directional flux contracts accept concrete inputs through compilation."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("case", ["immutable", "wrong-Zonal", "wrong-Meridional"])
def test_directional_transport_contract(tmp_path: Path, case: str) -> None:
    """Batch valid signatures while checking each direction rejects wrong State."""
    valid = case == "immutable"
    source = """from veris.model.types import State
from veris.model.configuration import Configuration
from jax import Array
from numpy import float64
from numpy.typing import NDArray
from veris.physics.transport.advection import calc_ZonalFlux, calc_MeridionalFlux
from veris.model.physical_constants import PhysicalConstants
"""
    for direction in ("Zonal", "Meridional"):
        state_type = "str" if case == f"wrong-{direction}" else "State"
        source += f"""
def evaluate_{direction}(state: {state_type}, constants: Configuration, field: NDArray[float64]) -> Array:
    return calc_{direction}Flux(state, constants, PhysicalConstants(), field, field)
"""
    root = Path(__file__).resolve().parents[3]
    path = tmp_path / "transport_contract.py"
    path.write_text(source)
    result = subprocess.run(
        ["ty", "check", "--extra-search-path", str(root), str(path)],
        cwd=root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostic = result.stdout + result.stderr
    if valid:
        assert result.returncode == 0, diagnostic[-2000:]
    else:
        assert result.returncode != 0, "ERROR invalid transport state accepted"
        assert "invalid-argument-type" in diagnostic, diagnostic[-2000:]


def test_boolean_ocean_masks_remain_valid_inputs(tmp_path: Path) -> None:
    """Typing must retain existing boolean-mask support in bulk and halo APIs."""
    root = Path(__file__).resolve().parents[3]
    source = """from jax import Array
from numpy import bool_, float64
from numpy.typing import NDArray
from veris.grid.halos import fill_overlap
from veris.physics.thermodynamics.heat_flux_cesm import dqnetdt
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants

def evaluate(conf: Configuration, phys: PhysicalConstants, mask: NDArray[bool_], field: NDArray[float64]) -> tuple[Array, Array, Array]:
    fill_overlap(mask, conf)
    return dqnetdt(conf, phys, mask, field, field, field, field, field, field, field)
"""
    path = tmp_path / "boolean_mask.py"
    path.write_text(source)
    result = subprocess.run(
        ["ty", "check", "--extra-search-path", str(root), str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
