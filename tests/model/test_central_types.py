"""All shared schemas have one owner and survive independent module imports."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module,name",
    [
        ("configuration", "Configuration"),
        ("physical_constants", "PhysicalConstants"),
        ("variables", "Variable"),
        ("diagnostics", "Diagnostics"),
    ],
)
def test_registry_types_have_local_owners(module: str, name: str) -> None:
    cls = getattr(importlib.import_module(f"veris.model.{module}"), name)
    assert cls.__module__ == f"veris.model.{module}"
    assert not hasattr(importlib.import_module("veris.model.types"), name)


def test_precision_is_setting_metadata_without_a_base_class() -> None:
    from veris.model import configuration, types

    assert isinstance(configuration.SETTINGS["dtype"], types.Parameter)
    assert not hasattr(configuration, "PRECISION")
    assert not hasattr(types, "Precision")
    assert not hasattr(types, "PRECISION")


def test_ocean_geometry_has_explicit_name() -> None:
    from veris.grid import geometry

    assert hasattr(geometry, "OceanGeometry")
    assert not hasattr(geometry, "Geometry")


@pytest.mark.parametrize(
    "first",
    ["types", "configuration", "physical_constants", "variables", "metadata"],
)
def test_registry_import_order_and_frozen_defaults(first: str) -> None:
    """Fresh interpreters expose cycles hidden by an already-imported test suite."""
    source = f"""
import importlib
importlib.import_module('veris.model.{first}')
from veris.model.types import State
from veris.grid.geometry import OceanGeometry
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
assert Configuration().deltatDyn == 86400
assert PhysicalConstants().rhoIce == 900
assert State.__module__ == 'veris.model.types'
assert OceanGeometry.__module__ == 'veris.grid.geometry'
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_registries_share_parameter_metadata() -> None:
    from veris.model import configuration, physical_constants, types

    assert hasattr(types, "Parameter")
    assert types.Parameter.__module__ == "veris.model.types"
    for registry in (configuration.SETTINGS, physical_constants.PHYSICALCONSTANTS):
        assert all(type(value) is types.Parameter for value in registry.values())
    assert not hasattr(configuration, "Setting")
    assert not hasattr(physical_constants, "PhysicalConstant")
