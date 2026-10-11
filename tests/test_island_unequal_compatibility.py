"""Original island compatibility retains independent physical timesteps."""

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import veris
from veris.setups import island


@pytest.mark.parametrize("wrapper", ["step_with_diagnostics", "step", "compiled_step"])
def test_island_unequal_timesteps_match_frozen_original(wrapper: str) -> None:
    """Check native arrays against archived original Dyn600/Therm3600 science."""
    state, conf, phys = island.initialize(
        4,
        6,
        settings_overrides={
            "dtype": "float64",
            "nEVPsteps": 2,
            "deltatDyn": 600,
            "deltatTherm": 3600,
        },
    )
    # Call first, so actual RED is the migrated wrapper's equality restriction,
    # not a missing fixture/path/schema failure.
    result = getattr(island, wrapper)(state, conf, phys, 80.0)
    numerical, diagnostics = (
        result if wrapper == "step_with_diagnostics" else (result, None)
    )
    assert conf.deltatDyn == 600 and conf.deltatTherm == 3600
    fixture = (
        Path(__file__).with_name("reference_data") / "island_unequal_legacy_v1.npz"
    )
    metadata = json.loads(fixture.with_suffix(".json").read_text())
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["archive_sha256"]
    )
    assert metadata["source_commit"] == "fff4cd4e08613e8ec2359e316c94435e81682400"
    with np.load(fixture, allow_pickle=False) as original:
        assert len(original.files) == 75
        for prefix, record in (("state", numerical), ("diagnostics", diagnostics)):
            if record is None:
                continue
            for item in fields(record):
                actual = np.asarray(getattr(record, item.name))
                expected = original[f"{prefix}.{item.name}"]
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                assert np.isfinite(actual).all()
                error = np.abs(actual - expected)
                assert np.all(error <= 2e-11 * (1 + np.abs(expected))), (
                    f"ERROR {wrapper} {prefix}.{item.name}: max_abs={error.max():.6g}"
                )


def test_public_coupled_execution_still_rejects_unequal_timesteps() -> None:
    """The legacy exception must not weaken the new lifecycle's time contract."""
    state, conf, phys = island.initialize(
        4, 6, settings_overrides={"deltatDyn": 600, "deltatTherm": 3600}
    )
    with pytest.raises(ValueError, match="coupled physics requires equal timesteps"):
        veris.advance(state, conf, phys)
    with pytest.raises(ValueError, match="equal dynamics/thermodynamic timesteps"):
        veris.Experiment.builtin(
            "island",
            grid=veris.Grid(4, 6),
            settings={"deltatDyn": 600, "deltatTherm": 3600},
        )
