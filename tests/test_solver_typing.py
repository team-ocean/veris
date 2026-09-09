"""Static solver contracts preserve iteration types and five-field returns."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("valid", [True, False], ids=["valid", "invalid"])
@pytest.mark.parametrize(
    "contract", ["iteration-count", "evp-result", "dispatcher-result"]
)
def test_solver_contract(tmp_path: Path, valid: bool, contract: str) -> None:
    """Reject fractional substep counts and incorrectly sized solver results."""
    source = """from dataclasses import dataclass
from jax import Array
from veris._solver_types import EVPState, IceVelocityState, IceVelocitySettings
from veris.evp_solver import evp_solver
from veris.dynsolver import IceVelocities

from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants

@dataclass(frozen=True)
class Constants:
    use_coastline: bool
    sideDragU0: float
    noSlip: bool
    secondOrderBC: bool
    deltaMin: float
    pressReplFac: float
    cDragMin: float
    basalDragSmoothing: float
    basalDragMinArea: float
    use_sharding: bool
    printEvpResidual: bool
    computeEvpResidual: bool
    useAdaptiveEVP: bool
    aEVPalphaMin: float
    aEVPmassMin: float
    aEVPcStar: float
    evpStressRelaxation: float
    evpShearRelaxation: float
    recip_deltatDyn: float
    deltatDyn: float
    aEvpCoeff: float
    evpAlpha: float
    evpBeta: float
    nEVPsteps: COUNT_TYPE

def evaluate(state: STATE_TYPE, constants: SETTINGS_TYPE, phys: PhysicalConstants) -> RETURN_TYPE:
    return SOLVER(state, constants, phys)
"""
    dispatcher = contract == "dispatcher-result"
    source = source.replace(
        "STATE_TYPE", "IceVelocityState" if dispatcher else "EVPState"
    )
    source = source.replace(
        "SETTINGS_TYPE", "IceVelocitySettings" if dispatcher else "Constants"
    )
    source = source.replace("SOLVER", "IceVelocities" if dispatcher else "evp_solver")
    count_type = "float" if contract == "iteration-count" and not valid else "int"
    source = source.replace("COUNT_TYPE", count_type)
    result_type = (
        "tuple[Array, Array]"
        if contract != "iteration-count" and not valid
        else "tuple[Array, Array, Array, Array, Array]"
    )
    source = source.replace("RETURN_TYPE", result_type)
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "solver_contract.py"
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
        assert result.returncode != 0, "ERROR invalid solver contract accepted"
        expected = (
            "invalid-argument-type"
            if contract == "iteration-count"
            else "invalid-return-type"
        )
        assert expected in diagnostic, diagnostic[-2000:]
