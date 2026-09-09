"""Static contracts must accept immutable states and reject invalid field types."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("valid", [True, False], ids=["immutable", "wrong-state"])
def test_concrete_mass_contract(tmp_path: Path, valid: bool) -> None:
    """Accept the model State and preserve the public compiled signature."""
    root = Path(__file__).resolve().parents[1]
    source = """from jax import Array
from veris.area_mass import SeaIceMass
from veris.state import State
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants

def evaluate(ice: STATE_TYPE, constants: PhysicalConstants) -> tuple[Array, Array, Array]:
    return SeaIceMass(ice, Settings(), constants)
""".replace("STATE_TYPE", "State" if valid else "str")
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
        assert result.returncode != 0, "ERROR invalid state accepted by typed kernel"
        assert "invalid-argument-type" in diagnostic, diagnostic[-2000:]


def test_mutable_static_settings_are_rejected(tmp_path: Path) -> None:
    """JIT static arguments must be hashable as well as having numeric fields."""
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "unhashable.py"
    path.write_text("""from dataclasses import dataclass
from veris.state import State
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.area_mass import SeaIceMass

@dataclass
class Constants:
    rhoIce: float
    rhoSnow: float

def invalid(state: State) -> None:
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


@pytest.mark.parametrize(
    "module_name",
    [
        "advection",
        "area_mass",
        "averaging",
        "clean_up",
        "dynamics_routines",
        "dynsolver",
        "evp_solver",
        "fill_overlap",
        "freedrift_solver",
        "growth",
        "ocean_stress",
        "solve4temp",
    ],
)
def test_kernels_use_concrete_model_annotations(module_name: str) -> None:
    """Every state/configuration kernel argument uses the initialized model class."""
    import importlib
    import inspect
    from typing import get_type_hints

    from veris.configuration import Settings
    from veris.state import State

    module = importlib.import_module(f"veris.{module_name}")
    checked = 0
    for function in vars(module).values():
        if (
            not callable(function)
            or getattr(function, "__module__", None) != module.__name__
        ):
            continue
        function = inspect.unwrap(function)
        annotations = get_type_hints(function)
        for argument, expected in (("vs", State), ("sett", Settings)):
            if argument in inspect.signature(function).parameters:
                assert annotations.get(argument) is expected, (
                    f"ERROR {module_name}.{function.__name__} {argument} must use {expected.__name__}"
                )
                checked += 1
    assert checked, f"ERROR no state/settings arguments checked in {module_name}"
