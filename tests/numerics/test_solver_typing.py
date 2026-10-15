"""Static solver contracts preserve iteration types and five-field returns."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "valid,contract",
    [
        (True, "both-results"),
        (False, "iteration-count"),
        (False, "evp-result"),
        (False, "dispatcher-result"),
    ],
)
def test_solver_contract(tmp_path: Path, valid: bool, contract: str) -> None:
    """Reject fractional substep counts and incorrectly sized solver results."""
    source = """from jax import Array
from veris.model.types import State
from veris.numerics.evp import evp_solver
from veris.numerics.momentum import IceVelocities

from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants

settings = Configuration(nEVPsteps=10)

def evaluate(state: State, constants: Configuration, phys: PhysicalConstants) -> RETURN_TYPE:
    return SOLVER(state, constants, phys)
"""
    dispatcher = contract == "dispatcher-result"
    source = source.replace("SOLVER", "IceVelocities" if dispatcher else "evp_solver")
    if contract == "iteration-count":
        count = "1.5" if not valid else "10"
        source += f"\nsettings = Configuration(nEVPsteps={count})\n"
    result_type = (
        "tuple[Array, Array]"
        if contract != "iteration-count" and not valid
        else "tuple[Array, Array, Array, Array, Array]"
    )
    source = source.replace("RETURN_TYPE", result_type)
    if valid:
        source += """
def dispatch(state: State, constants: Configuration, phys: PhysicalConstants) -> tuple[Array, Array, Array, Array, Array]:
    return IceVelocities(state, constants, phys)
"""
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "solver_contract.py"
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
        assert result.returncode != 0, "ERROR invalid solver contract accepted"
        expected = (
            "invalid-argument-type"
            if contract == "iteration-count"
            else "invalid-return-type"
        )
        assert expected in diagnostic, diagnostic[-2000:]
