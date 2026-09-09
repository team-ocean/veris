"""Static contracts must accept immutable states and reject invalid field types."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("valid", [True, False], ids=["immutable", "wrong-fields"])
def test_structural_mass_contract(tmp_path: Path, valid: bool) -> None:
    """Catch writable-only protocols and loss of the public kernel signature."""
    root = Path(__file__).resolve().parents[1]
    source = """from typing import NamedTuple
from jax import Array
from veris.area_mass import SeaIceMass
from veris._typing import ThicknessState
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants

class Ice(NamedTuple):
    hIceMean: Array
    hSnowMean: FIELD_TYPE

class Constants(NamedTuple):
    rhoIce: float
    rhoSnow: float

def evaluate(ice: Ice, constants: PhysicalConstants) -> tuple[Array, Array, Array]:
    return SeaIceMass(ice, Settings(), constants)
""".replace("FIELD_TYPE", "Array" if valid else "str")
    path = tmp_path / "contract.py"
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
        assert result.returncode != 0, (
            "ERROR invalid snow field accepted by typed kernel"
        )
        assert "invalid-argument-type" in diagnostic, diagnostic[-2000:]


def test_mutable_static_settings_are_rejected(tmp_path: Path) -> None:
    """JIT static arguments must be hashable as well as having numeric fields."""
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "unhashable.py"
    path.write_text("""from dataclasses import dataclass
from veris._typing import ThicknessState
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.area_mass import SeaIceMass

@dataclass
class Constants:
    rhoIce: float
    rhoSnow: float

def invalid(state: ThicknessState) -> None:
    SeaIceMass(state, Constants(900., 330.), PhysicalConstants())
""")
    result = subprocess.run(
        ["ty", "check", "--extra-search-path", str(root), str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    diagnostic = result.stdout + result.stderr
    assert result.returncode != 0, "ERROR mutable unhashable static settings accepted"
    assert "invalid-argument-type" in diagnostic, diagnostic[-2000:]
