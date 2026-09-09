"""Bulk and thermodynamic APIs retain useful contracts after JIT compilation."""

import subprocess
import sys
from pathlib import Path

import pytest


def _check_contract(
    tmp_path: Path, source: str, expected: tuple[str, str] | None
) -> None:
    """Check the public call, requiring a specific reason for invalid inputs."""
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "heat_flux_contract.py"
    path.write_text(source)
    result = subprocess.run(
        [
            "ty",
            "check",
            "--python",
            sys.executable,
            "--extra-search-path",
            str(root),
            str(path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    diagnostic = result.stdout + result.stderr
    if expected is None:
        assert result.returncode == 0, diagnostic[-2000:]
    else:
        assert result.returncode != 0, "ERROR invalid heat-flux contract accepted"
        for fragment in expected:
            assert fragment in diagnostic, diagnostic[-2000:]


@pytest.mark.parametrize(
    "case", ["immutable", "missing-settings", "wrong-coefficient", "wrong-arity"]
)
def test_bulk_heat_flux_contract(tmp_path: Path, case: str) -> None:
    """Accept initialized configuration, NumPy fields, and scalar helpers."""
    source = """import numpy as np
from jax import Array
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris.heat_flux_CESM import dqnetdt, get_press_levs, qsat, qsat_august_eqn, cdn
from veris.heat_flux_MITgcm import bulkf_formula_lanl

def evaluate(sett: Settings, phys: PhysicalConstants) -> tuple[Array, Array, Array]:
    field = np.ones((2, 3))
    return dqnetdt(sett, phys, field, field, field, field, field, field, field, field)

def pressure_levels() -> Array:
    return get_press_levs(np.ones((2, 3)), np.ones(4), np.ones(4))

def scalar_helpers(phys: PhysicalConstants) -> tuple[Array, Array, Array]:
    return qsat(phys, 275.0), qsat_august_eqn(phys, 100000.0, 275.0), cdn(phys, 5.0)
"""
    expected = None
    if case == "missing-settings":
        source = source.replace("dqnetdt(sett, phys,", "dqnetdt(None, phys,")
        expected = ("invalid-argument-type", "Settings")
    elif case == "wrong-coefficient":
        source += '\nconstants = PhysicalConstants(cpdair="invalid")\n'
        expected = ("invalid-argument-type", "float")
    elif case == "wrong-arity":
        source += """
def wrong_length(sett: Settings, phys: PhysicalConstants, field: Array) -> tuple[Array, Array]:
    return bulkf_formula_lanl(sett, phys, field, field, field, field, field, field)
"""
        expected = ("invalid-return-type", "tuple of length 9")
    _check_contract(tmp_path, source, expected)


@pytest.mark.parametrize(
    "case", ["immutable", "wrong-atmosphere", "wrong-category-count", "wrong-arity"]
)
def test_thermodynamic_heat_flux_contract(tmp_path: Path, case: str) -> None:
    """Check concrete State, configuration, and thermodynamic result types."""
    source = f"""import numpy as np
from jax import Array
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants
from veris._thermodynamic_types import GrowthResult
from veris.state import State
from veris.growth import Growth
from veris.solve4temp import solve4temp

def surface_fluxes(state: {"str" if case == "wrong-atmosphere" else "State"}, sett: Settings, phys: PhysicalConstants) -> tuple[Array, Array, Array, Array, Array]:
    field = np.ones((2, 3))
    return solve4temp(state, sett, phys, field, field, field, field)

def growth(state: State, sett: Settings, phys: PhysicalConstants) -> {"tuple[Array, Array]" if case == "wrong-arity" else "GrowthResult"}:
    return Growth(state, sett, phys)
"""
    expected = None
    if case == "wrong-atmosphere":
        expected = ("invalid-argument-type", "State")
    elif case == "wrong-category-count":
        source += "\nsettings = Settings(nITC=1.5)\n"
        expected = ("invalid-argument-type", "int")
    elif case == "wrong-arity":
        expected = ("invalid-return-type", "tuple of length 11")
    _check_contract(tmp_path, source, expected)


def test_height_uses_initialized_physical_constants(tmp_path: Path) -> None:
    """The height helper consumes the same initialized physical constants."""
    _check_contract(
        tmp_path,
        """from jax import Array
from veris.physical_constants import PhysicalConstants
from veris.heat_flux_CESM import compute_z_level
def evaluate(phys: PhysicalConstants, t: Array, q: Array, ph: Array) -> Array:
    return compute_z_level(phys, t, q, ph)
""",
        None,
    )
