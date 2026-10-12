"""Public lifecycle and compatibility wrappers match frozen original trajectories."""

import hashlib
import json
from dataclasses import fields
from importlib import import_module
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

import veris

_ROOT = Path(__file__).resolve().parents[1] / "reference_data"


def _compare(actual: Any, reference: np.ndarray, name: str) -> None:
    """Use established dtype-specific tolerances and compact error reporting."""
    actual = np.asarray(actual)
    assert actual.shape == reference.shape and actual.dtype == reference.dtype, (
        f"ERROR {name} storage or dtype differs"
    )
    tolerance = 3e-5 if reference.dtype == np.dtype("float32") else 2e-11
    error = np.abs(actual - reference)
    assert np.all(error <= tolerance * (1 + np.abs(reference))), (
        f"ERROR {name}: max_abs={error.max():.6g}"
    )


@pytest.mark.parametrize("case_index", range(7))
def test_public_and_compatibility_trajectories_match_immutable_reference(
    case_index: int,
) -> None:
    """Every State and diagnostic field anchors original ordering and startup."""
    manifest = json.loads((_ROOT / "lifecycle_legacy_v1.json").read_text())
    archive = _ROOT / "lifecycle_legacy_v1.npz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    )
    case = manifest["cases"][case_index]
    preset, name = case["preset"], case["name"]
    forcing = (
        veris.Forcing(
            constant={"Qsw": 0}, series={"Qnet": np.array(case["forcing"]["Qnet"])}
        )
        if preset == "island"
        else veris.Forcing()
    )
    experiment = veris.Experiment.builtin(
        preset,
        grid=veris.Grid(*case["grid"]),
        settings=case["settings"],
        forcing=forcing,
        steps=3,
    )
    state, conf, phys = veris.initialize_experiment(experiment)
    module = import_module(
        "veris.setups.island"
        if preset == "island"
        else f"veris.setups.run_{'dyn' if preset == 'dynamics' else 'growth'}"
    )
    legacy, _, _ = (
        module.initialize(settings_overrides=case["settings"])
        if preset == "growth"
        else module.initialize(*case["grid"], settings_overrides=case["settings"])
    )
    transition = jax.jit(module.step_with_diagnostics, static_argnums=(1, 2))
    with (
        np.load(archive, allow_pickle=False) as reference,
        veris.Simulation.initialize(experiment) as simulation,
    ):
        for step in range(4):
            if step in (0, 1, 3):
                for item in fields(state):
                    expected = reference[f"{name}.state.{step}.{item.name}"]
                    for label, actual in (
                        ("pure", state),
                        ("legacy", legacy),
                        ("host", simulation.state),
                    ):
                        _compare(
                            getattr(actual, item.name),
                            expected,
                            f"{name}.{label}.{step}.{item.name}",
                        )
            if step == 3:
                break
            prescribed = forcing.at(step)
            state, diagnostic = veris.advance(
                state, conf, phys, prescribed, physics=experiment.physics
            )
            legacy, legacy_diagnostic = (
                transition(legacy, conf, phys, prescribed["Qnet"])
                if preset == "island"
                else transition(legacy, conf, phys)
            )
            result = simulation.step()
            if step + 1 in (1, 3):
                for item in fields(diagnostic):
                    expected = reference[f"{name}.diagnostics.{step + 1}.{item.name}"]
                    for label, actual in (
                        ("pure", diagnostic),
                        ("legacy", legacy_diagnostic),
                        ("host", result.diagnostics),
                    ):
                        _compare(
                            getattr(actual, item.name),
                            expected,
                            f"{name}.{label}.{step + 1}.{item.name}",
                        )
