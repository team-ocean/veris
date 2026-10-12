"""Pure timesteps and the public lifecycle match frozen original trajectories."""

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

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
def test_pure_and_host_trajectories_match_immutable_reference(
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
    # These archives predate restoration of the standalone notebook setup.
    # Retain their corrected thermodynamic equations and every numerical oracle;
    # explicitly adapt only the intentionally restored inactive startup fields.
    settings = dict(case["settings"])
    restored_defaults: dict[str, float] = {}
    if preset == "growth":
        settings["useLegacyGrowthAreaWeights"] = False
        restored_defaults = dict.fromkeys(
            (
                "dxG",
                "dyG",
                "dxU",
                "dyU",
                "dxV",
                "dyV",
                "recip_dxC",
                "recip_dyC",
                "recip_dxU",
                "recip_dyU",
                "recip_dxV",
                "recip_dyV",
                "rAz",
                "recip_rA",
                "recip_rAu",
                "recip_rAv",
            ),
            1.0,
        )
    elif preset == "dynamics":
        restored_defaults = {"theta": 273.15, "ATemp": 273.15}
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
        settings=settings,
        forcing=forcing,
        steps=3,
    )
    state, conf, phys = veris.initialize_experiment(experiment)
    with (
        np.load(archive, allow_pickle=False) as reference,
        veris.Simulation.initialize(experiment) as simulation,
    ):
        for step in range(4):
            if step in (0, 1, 3):
                for item in fields(state):
                    expected = reference[f"{name}.state.{step}.{item.name}"]
                    if item.name in restored_defaults:
                        assert np.all(expected == restored_defaults[item.name])
                        expected = np.zeros_like(expected)
                    for label, actual in (
                        ("pure", state),
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
            result = simulation.step()
            if step + 1 in (1, 3):
                for item in fields(diagnostic):
                    expected = reference[f"{name}.diagnostics.{step + 1}.{item.name}"]
                    for label, actual in (
                        ("pure", diagnostic),
                        ("host", result.diagnostics),
                    ):
                        _compare(
                            getattr(actual, item.name),
                            expected,
                            f"{name}.{label}.{step + 1}.{item.name}",
                        )
