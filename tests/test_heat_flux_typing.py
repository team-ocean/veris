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
    """Accept minimal immutable constants, NumPy fields, and scalar helpers."""
    coefficient_type = "str" if case == "wrong-coefficient" else "float"
    state_field = (
        "unused: float" if case == "missing-settings" else "settings: Constants"
    )
    source = f"""from typing import NamedTuple
import numpy as np
from jax import Array
from veris._bulk_types import BulkState, LANLFluxSettings
from veris.heat_flux_CESM import dqnetdt, get_press_levs, qsat, qsat_august_eqn, cdn
from veris.heat_flux_MITgcm import bulkf_formula_lanl

class Constants(NamedTuple):
    ce: float
    ch: float
    cpdair: {coefficient_type}
    latvap: float
    stefBoltz: float
    umin_o: float

class State(NamedTuple):
    {state_field}

def evaluate(state: State) -> tuple[Array, Array, Array]:
    field = np.ones((2, 3))
    return dqnetdt(state, field, field, field, field, field, field, field, field)

def pressure_levels() -> Array:
    return get_press_levs(np.ones((2, 3)), np.ones(4), np.ones(4))

def scalar_helpers() -> tuple[Array, Array, Array]:
    return qsat(275.0), qsat_august_eqn(100000.0, 275.0), cdn(5.0)
"""
    expected = None
    if case == "missing-settings":
        expected = ("invalid-argument-type", "member `settings` is not defined")
    elif case == "wrong-coefficient":
        expected = ("invalid-argument-type", "member `cpdair` is incompatible")
    elif case == "wrong-arity":
        source += """
def wrong_length(state: BulkState[LANLFluxSettings], field: Array) -> tuple[Array, Array]:
    return bulkf_formula_lanl(state, field, field, field, field, field, field)
"""
        expected = ("invalid-return-type", "tuple of length 9")
    _check_contract(tmp_path, source, expected)


@pytest.mark.parametrize(
    "case", ["immutable", "wrong-atmosphere", "wrong-category-count", "wrong-arity"]
)
def test_thermodynamic_heat_flux_contract(tmp_path: Path, case: str) -> None:
    """Require atmospheric arrays and integer category counts in coupled growth."""
    fields = ["ATemp", "LWdown", "SWdown", "aqh", "fCori", "wSpeed"]
    atmosphere = "\n".join(
        f"    {name}: {'str' if case == 'wrong-atmosphere' and name == 'aqh' else 'Array'}"
        for name in fields
    )
    constants = [
        "Area_reg",
        "McPheeTaperFac",
        "celsius2K",
        "cpWater",
        "deltatTherm",
        "dtempFrz_dS",
        "hIce_reg",
        "lhFusion",
        "nITC",
        "recip_deltatTherm",
        "recip_h0",
        "recip_h0_south",
        "recip_nITC",
        "recip_rhoSea",
        "rhoFresh",
        "rhoFresh2rhoSnow",
        "rhoIce",
        "rhoIce2rhoFresh",
        "rhoIce2rhoSnow",
        "rhoSea",
        "rhoSnow",
        "saltIce_ref",
        "stantonNr",
        "tempFrz",
        "uStarBase",
        "cpAir",
        "dalton",
        "dryIceAlb",
        "dryIceAlb_south",
        "drySnowAlb",
        "drySnowAlb_south",
        "hCut",
        "iceConduct",
        "iceEmiss",
        "lhSublim",
        "minLWdown",
        "minTAir",
        "minTIce",
        "rhoAir",
        "shortwave",
        "snowConduct",
        "snowEmiss",
        "stefBoltz",
        "wSpeedMin",
        "wetAlbTemp",
        "wetIceAlb",
        "wetIceAlb_south",
        "wetSnowAlb",
        "wetSnowAlb_south",
    ]
    declarations = "\n".join(
        f"    {name}: {'int' if name == 'nITC' and case != 'wrong-category-count' else 'float'}"
        for name in constants
    )
    source = f"""from typing import NamedTuple
import numpy as np
from jax import Array
from veris._thermodynamic_types import GrowthState, GrowthResult
from veris.growth import Growth
from veris.solve4temp import solve4temp

class Atmosphere(NamedTuple):
{atmosphere}

class Constants(NamedTuple):
{declarations}

def surface_fluxes(state: Atmosphere, constants: Constants) -> tuple[Array, Array, Array, Array, Array]:
    field = np.ones((2, 3))
    return solve4temp(state, constants, field, field, field, field)

def growth(state: GrowthState, constants: Constants) -> {"tuple[Array, Array]" if case == "wrong-arity" else "GrowthResult"}:
    return Growth(state, constants)
"""
    expected = None
    if case == "wrong-atmosphere":
        expected = ("invalid-argument-type", "member `aqh` is incompatible")
    elif case == "wrong-category-count":
        expected = ("invalid-argument-type", "member `nITC` is incompatible")
    elif case == "wrong-arity":
        expected = ("invalid-return-type", "tuple of length 11")
    _check_contract(tmp_path, source, expected)


def test_height_settings_need_not_be_hashable(tmp_path: Path) -> None:
    """The uncompiled height helper accepts mutable atmospheric constants."""
    _check_contract(
        tmp_path,
        """from dataclasses import dataclass
from jax import Array
from veris.heat_flux_CESM import compute_z_level
@dataclass
class Constants:
    grav: float
    radius: float
    rdair: float
    zvir: float
def evaluate(t: Array, q: Array, ph: Array) -> Array:
    return compute_z_level(Constants(9.81, 6371000., 287., 0.61), t, q, ph)
""",
        None,
    )
