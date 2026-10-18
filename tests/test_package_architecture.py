"""Canonical package ownership, current public API and cold import boundaries."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_public_facade_exports_keep_canonical_identity() -> None:
    """Public exports retain their canonical objects, including launch APIs."""
    import veris

    owners = {
        "DistributedExecution": "veris.model.experiment",
        "Execution": "veris.model.experiment",
        "Experiment": "veris.model.experiment",
        "Grid": "veris.model.experiment",
        "Forcing": "veris.model.forcing",
        "OceanGeometry": "veris.grid.geometry",
        "RunResult": "veris.execution.simulation",
        "Simulation": "veris.execution.simulation",
        "advance": "veris.execution.advance",
        "diagnose_surface_albedo": "veris.numerics.surface_properties",
        "initialize_experiment": "veris.execution.experiment_initialization",
        "step": "veris.execution.rollout",
        "run_experiment": "veris.execution.runner",
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
        "veris.cli",
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
        check=False,
    )
    assert result.returncode == 0, f"ERROR cold import {first}: {result.stderr[-2000:]}"


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


def test_package_root_has_only_metadata_and_public_api() -> None:
    """Implementations belong to their domains rather than old flat paths."""
    root = Path(__file__).resolve().parents[1] / "veris"
    assert {path.name for path in root.glob("*.py")} == {"__init__.py", "_version.py"}


def test_model_schemas_have_explicit_owners() -> None:
    """Grid geometry and array inputs do not leak through the State schema."""
    types = importlib.import_module("veris.model.types")
    assert not {"OceanGeometry", "ArrayInput", "MaskInput"} & set(vars(types))


@pytest.mark.parametrize(
    "module,names",
    [
        (
            "veris.setups.island",
            ("initialize", "step", "step_with_diagnostics", "compiled_step"),
        ),
        (
            "veris.setups.run_dyn",
            ("initialize", "step", "step_with_diagnostics", "compiled_step"),
        ),
        (
            "veris.setups.run_growth",
            ("initialize", "step", "step_with_diagnostics", "compiled_step"),
        ),
        (
            "veris.setups.run_parallel",
            (
                "create_mesh",
                "distributed_options",
                "parse_args",
                "run_timed",
                "remove_halos",
                "gather_output",
                "save_output",
            ),
        ),
        (
            "veris.io.cli",
            ("output_settings", "prepare_output_paths", "make_output", "save_final"),
        ),
        ("veris.grid.halos", ("make_sharded_fill_overlap",)),
        ("veris.model.experiment", ("_InitializationExperiment",)),
        ("veris.setups.presets", ("_initialize_adapter", "_UnknownScenarioError")),
    ],
)
def test_setups_and_cli_do_not_export_obsolete_execution_helpers(
    module: str, names: tuple[str, ...]
) -> None:
    """Execution and storage APIs belong to their canonical domain owners."""
    implementation = importlib.import_module(module)
    assert not set(names) & set(vars(implementation))
