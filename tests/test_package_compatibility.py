"""Canonical domain ownership preserves every previous module and public object.

These migration regressions fail before source moves; numerical equivalence is
owned by the unchanged all-field references and equation/gradient suites.
"""

import importlib
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

MIGRATIONS = {
    "veris._typing": "veris.model.types",
    "veris._metadata": "veris.model.metadata",
    "veris.configuration": "veris.model.configuration",
    "veris.physical_constants": "veris.model.physical_constants",
    "veris.variables": "veris.model.variables",
    "veris.diagnostics": "veris.model.diagnostics",
    "veris.experiment": "veris.model.experiment",
    "veris.experiment_file": "veris.model.experiment_file",
    "veris.forcing": "veris.model.forcing",
    "veris.fill_overlap": "veris.grid.halos",
    "veris.averaging": "veris.grid.averaging",
    "veris.global_sum": "veris.grid.reductions",
    "veris.model_regions": "veris.grid.regions",
    "veris.setups.ocean": "veris.grid.ocean",
    "veris.dynamics": "veris.physics.dynamics.driver",
    "veris.dynamics_routines": "veris.physics.dynamics.rheology",
    "veris.area_mass": "veris.physics.dynamics.area_mass",
    "veris.ocean_stress": "veris.physics.dynamics.ocean_stress",
    "veris.advection": "veris.physics.transport.advection",
    "veris.clean_up": "veris.physics.transport.cleanup",
    "veris.growth": "veris.physics.thermodynamics.growth",
    "veris.heat_flux_CESM": "veris.physics.thermodynamics.heat_flux_cesm",
    "veris.heat_flux_MITgcm": "veris.physics.thermodynamics.heat_flux_mitgcm",
    "veris._ad": "veris.numerics.autodiff",
    "veris.dynsolver": "veris.numerics.momentum",
    "veris.evp_solver": "veris.numerics.evp",
    "veris.freedrift_solver": "veris.numerics.freedrift",
    "veris.solve4temp": "veris.numerics.surface_temperature",
    "veris.initialization": "veris.execution.initialization",
    "veris.model_initialization": "veris.execution.experiment_initialization",
    "veris.model_physics": "veris.execution.advance",
    "veris.integration": "veris.execution.rollout",
    "veris.integration_output": "veris.execution.rollout_output",
    "veris.simulation": "veris.execution.simulation",
    "veris.simulation_io": "veris.io.lifecycle",
    "veris.restart": "veris.io.restart",
    "veris.restart_arrays": "veris.io.restart_arrays",
    "veris.presets": "veris.setups.presets",
}


@pytest.mark.parametrize("legacy,canonical", MIGRATIONS.items())
def test_legacy_import_is_canonical_module(legacy: str, canonical: str) -> None:
    """Patches, private helpers and JAX classes share one implementation owner."""
    previous = importlib.import_module(legacy)
    current = importlib.import_module(canonical)
    assert previous is current, f"ERROR duplicated module {legacy}"
    assert current.__name__ == canonical
    assert current.__file__ is not None
    assert Path(current.__file__).relative_to(
        Path(__file__).resolve().parents[1]
    ) == Path(canonical.replace(".", "/") + ".py")


@pytest.mark.parametrize("legacy,canonical", MIGRATIONS.items())
def test_legacy_symbols_and_patches_share_globals(
    legacy: str, canonical: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old imports can patch globals consulted through the canonical module."""
    previous = importlib.import_module(legacy)
    current = importlib.import_module(canonical)
    marker = object()
    monkeypatch.setattr(previous, "_migration_identity_probe", marker, raising=False)
    assert getattr(current, "_migration_identity_probe", None) is marker
    for name in getattr(current, "__all__", ()):
        assert getattr(previous, name) is getattr(current, name)


@pytest.mark.parametrize(
    "legacy,name,canonical",
    [
        ("veris._typing", "State", "veris.model.types"),
        ("veris._typing", "Parameter", "veris.model.types"),
        ("veris._typing", "OceanGeometry", "veris.grid.geometry"),
        ("veris.configuration", "Configuration", "veris.model.configuration"),
        (
            "veris.physical_constants",
            "PhysicalConstants",
            "veris.model.physical_constants",
        ),
        ("veris.variables", "Variable", "veris.model.variables"),
        ("veris.diagnostics", "Diagnostics", "veris.model.diagnostics"),
        ("veris.experiment", "Experiment", "veris.model.experiment"),
        ("veris.forcing", "Forcing", "veris.model.forcing"),
    ],
)
def test_legacy_pickle_globals_resolve_canonical_class(
    legacy: str, name: str, canonical: str
) -> None:
    """Previously serialized GLOBAL names still resolve after class ownership moves."""
    previous = pickle.loads(f"c{legacy}\n{name}\n.".encode())
    current = getattr(importlib.import_module(canonical), name)
    assert previous is current
    assert current.__module__ == canonical


def test_public_facade_exports_keep_canonical_identity() -> None:
    """The existing root interface has the same names and underlying objects."""
    import veris

    owners = {
        "Execution": "veris.model.experiment",
        "Experiment": "veris.model.experiment",
        "Grid": "veris.model.experiment",
        "Forcing": "veris.model.forcing",
        "OceanGeometry": "veris.grid.geometry",
        "RunResult": "veris.execution.simulation",
        "Simulation": "veris.execution.simulation",
        "advance": "veris.execution.advance",
        "initialize_experiment": "veris.execution.experiment_initialization",
        "step": "veris.execution.rollout",
    }
    assert set(veris.__all__) == set(owners)
    for name, owner in owners.items():
        assert getattr(veris, name) is getattr(importlib.import_module(owner), name)


@pytest.mark.parametrize(
    "first",
    [
        "veris.model.configuration",
        "veris.grid.geometry",
        "veris.numerics.evp",
        "veris.execution.simulation",
        "veris.io.restart",
        "veris.setups.run_parallel",
    ],
)
def test_cold_import_orders_preserve_jax_configuration(first: str) -> None:
    """Fresh interpreters expose cycles; imports neither select devices nor set flags."""
    script = f"""
import importlib
import jax
before = (jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision)
def forbid_devices(*args, **kwargs):
    raise AssertionError('ERROR import enumerated JAX devices')
jax.devices = forbid_devices
importlib.import_module({first!r})
from veris import Experiment, OceanGeometry
from veris.model.types import State
assert Experiment.__module__ == 'veris.model.experiment'
assert OceanGeometry.__module__ == 'veris.grid.geometry'
assert State.__module__ == 'veris.model.types'
assert before == (jax.config.jax_enable_x64, jax.config.jax_default_matmul_precision)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"ERROR cold import {first}: {result.stderr[-2000:]}"


def test_geometry_and_array_contracts_have_single_owners() -> None:
    """Geometry/schema imports avoid cyclic ownership or duplicate classes."""
    previous = importlib.import_module("veris._typing")
    geometry = importlib.import_module("veris.grid.geometry")
    arrays = importlib.import_module("veris.model.arrays")
    assert previous.OceanGeometry is geometry.OceanGeometry
    assert previous.ArrayInput is arrays.ArrayInput
    assert previous.MaskInput is arrays.MaskInput


def test_canonical_packages_do_not_eagerly_import_execution() -> None:
    """Domain initializers contain only documentation, keeping dependencies explicit."""
    import ast

    root = Path(__file__).resolve().parents[1] / "veris"
    for namespace in (
        "model",
        "grid",
        "physics",
        "physics/dynamics",
        "physics/transport",
        "physics/thermodynamics",
        "numerics",
        "execution",
    ):
        statements = ast.parse((root / namespace / "__init__.py").read_text()).body
        assert all(
            isinstance(item, ast.Expr)
            and isinstance(item.value, ast.Constant)
            and isinstance(item.value.value, str)
            for item in statements
        )
