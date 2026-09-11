"""All shared schemas have one owner and survive independent module imports."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module,name",
    [
        ("configuration", "Setting"),
        ("configuration", "Configuration"),
        ("physical_constants", "PhysicalConstant"),
        ("physical_constants", "PhysicalConstants"),
        ("variables", "Variable"),
        ("diagnostics", "Diagnostics"),
    ],
)
def test_registry_types_have_local_owners(module: str, name: str) -> None:
    cls = getattr(importlib.import_module(f"veris.{module}"), name)
    assert cls.__module__ == f"veris.{module}"
    assert not hasattr(importlib.import_module("veris._typing"), name)


def test_precision_is_setting_metadata_without_a_base_class() -> None:
    from veris import _typing, configuration

    assert isinstance(configuration.SETTINGS["dtype"], configuration.Setting)
    assert not hasattr(configuration, "PRECISION")
    assert not hasattr(_typing, "Precision")
    assert not hasattr(_typing, "PRECISION")


def test_ocean_geometry_has_explicit_name() -> None:
    from veris import _typing

    assert hasattr(_typing, "OceanGeometry")
    assert not hasattr(_typing, "Geometry")


def test_shared_type_files_are_removed() -> None:
    root = Path(__file__).resolve().parents[1] / "veris"
    for name in (
        "state.py",
        "_bulk_types.py",
        "_solver_types.py",
        "_thermodynamic_types.py",
    ):
        assert not (root / name).exists(), f"redundant type definition file: {name}"


@pytest.mark.parametrize(
    "first",
    ["_typing", "configuration", "physical_constants", "variables", "_metadata"],
)
def test_registry_import_order_and_frozen_defaults(first: str) -> None:
    """Fresh interpreters expose cycles hidden by an already-imported test suite."""
    source = f"""
import importlib
importlib.import_module('veris.{first}')
from dataclasses import replace
from veris._typing import State, OceanGeometry
from veris.configuration import Setting
from veris.physical_constants import PhysicalConstant
from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants
assert Configuration().deltatDyn == 86400
assert PhysicalConstants().rhoIce == 900
assert replace(Configuration(), deltatDyn=600).recip_deltatDyn == 1/600
assert State.__module__ == OceanGeometry.__module__ == 'veris._typing'
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize(
    "module,name",
    [("configuration", "Configuration"), ("physical_constants", "PhysicalConstants")],
)
def test_configuration_types_live_with_their_registries(module: str, name: str) -> None:
    """The settings and physical constants classes are explicit local exceptions."""
    cls = getattr(importlib.import_module(f"veris.{module}"), name)
    assert cls.__module__ == f"veris.{module}"
    assert not hasattr(importlib.import_module("veris._typing"), name)
