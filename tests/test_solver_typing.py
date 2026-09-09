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
    source = """from jax import Array
from veris.state import State
from veris.evp_solver import evp_solver
from veris.dynsolver import IceVelocities

from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants

def evaluate(state: State, constants: Settings, phys: PhysicalConstants) -> RETURN_TYPE:
    return SOLVER(state, constants, phys)
"""
    dispatcher = contract == "dispatcher-result"
    source = source.replace("SOLVER", "IceVelocities" if dispatcher else "evp_solver")
    if contract == "iteration-count":
        count = "1.5" if not valid else "10"
        source += f"\nsettings = Settings(nEVPsteps={count})\n"
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
