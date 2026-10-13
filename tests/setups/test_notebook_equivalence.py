"""Standalone initialization and dynamics match jax_halo_exchange notebooks.

The frozen original source supplies initialization, configuration and numerical
outputs. Tests never regenerate their oracle from the implementation under test.
"""

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from veris import Experiment, Grid, Simulation, initialize_experiment
from veris.execution.advance import advance
from veris.model.types import State

_ROOT = Path(__file__).resolve().parents[1] / "reference_data"


@pytest.fixture(scope="module")
def notebook_reference() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load checksum-verified original outputs and their capture provenance."""
    manifest = json.loads((_ROOT / "standalone_notebooks_v1.json").read_text())
    archive = _ROOT / "standalone_notebooks_v1.npz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    )
    generator = _ROOT / "generate_standalone_notebooks_reference.py.txt"
    assert (
        hashlib.sha256(generator.read_bytes()).hexdigest()
        == manifest["generator_sha256"]
    )
    with np.load(archive, allow_pickle=False) as source:
        return manifest, {name: source[name] for name in source.files}


def _compare(actual: Any, expected: np.ndarray, label: str) -> None:
    """Report compact maximum scaled error instead of printing model arrays."""
    actual = np.asarray(actual)
    assert actual.shape == expected.shape, f"ERROR {label}: shape differs"
    error = np.abs(actual - expected) / (1 + np.abs(expected))
    assert np.isfinite(actual).all() and np.all(error <= 2e-11), (
        f"ERROR {label}: max_scaled_error={error.max():.6g} "
        f"at {np.unravel_index(error.argmax(), error.shape)}"
    )


@pytest.mark.parametrize("preset", ["growth", "dynamics"])
def test_all_notebook_startup_fields_and_shared_constants(
    preset: str,
    notebook_reference: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    """Catch inherited registry defaults changing the original experiment."""
    manifest, source = notebook_reference
    case = next(case for case in manifest["cases"] if case["name"] == preset)
    state, conf, phys = (
        initialize_experiment(Experiment.builtin("growth"))
        if preset == "growth"
        else initialize_experiment(
            Experiment.builtin("dynamics", grid=Grid(*case["grid"]))
        )
    )
    assert set(manifest["state_fields"]) == {item.name for item in fields(State)}
    for item in fields(State):
        expected = source[f"{preset}.state.0.{item.name}"]
        if preset == "growth":
            expected = np.broadcast_to(expected, state.hIceMean.shape)
        _compare(getattr(state, item.name), expected, f"{preset}.initial.{item.name}")
    for name, expected in case["common_settings_constants"].items():
        if name == "use_sharding":
            assert not conf.use_sharding  # notebook serial execution adaptation
            continue
        owner = conf if hasattr(conf, name) else phys
        assert getattr(owner, name) == expected, f"ERROR {preset}.{name} differs"
    assert conf.deltatTherm == case["timestep_seconds"]


def test_dynamics_physical_fields_follow_original_100_step_notebook(
    notebook_reference: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    """Catch changes in solver controls, winds, masks or the notebook sequence."""
    manifest, source = notebook_reference
    case = next(case for case in manifest["cases"] if case["name"] == "dynamics")
    definition = Experiment.builtin("dynamics", grid=Grid(*case["grid"]), steps=100)
    state, conf, phys = initialize_experiment(
        Experiment.builtin("dynamics", grid=Grid(*case["grid"]))
    )
    transition = jax.jit(
        advance, static_argnames=("configuration", "constants", "physics")
    )
    for step in range(1, 101):
        state, diagnostics = transition(state, conf, phys, physics="dynamics")
        if step in (1, 3, 100):
            for item in fields(State):
                # The maintained driver refreshes all halos; the notebook only
                # refreshes thickness/snow/area. Physical cells are the oracle.
                expected = source[f"dynamics.state.{step}.{item.name}"][2:-2, 2:-2]
                _compare(
                    getattr(state, item.name)[2:-2, 2:-2],
                    expected,
                    f"dynamics.{step}.{item.name}",
                )
            for name in ("OceanStressU", "OceanStressV"):
                _compare(
                    getattr(diagnostics, name)[2:-2, 2:-2],
                    source[f"dynamics.diagnostics.{step}.{name}"][2:-2, 2:-2],
                    f"dynamics.{step}.{name}",
                )
    with Simulation.initialize(definition) as model:
        result = model.run()
    for item in fields(State):
        _compare(
            getattr(result.state, item.name)[2:-2, 2:-2],
            source[f"dynamics.state.100.{item.name}"][2:-2, 2:-2],
            f"dynamics.host.100.{item.name}",
        )
