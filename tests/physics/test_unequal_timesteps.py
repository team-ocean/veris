"""Kernel composition retains the archived Dyn600/Therm3600 island science.

The public coupled lifecycle requires a single timestep. This archive instead
checks the original independent dynamics and thermodynamic intervals directly
at their numerical owners, including all State and coupling Diagnostics arrays.
"""

import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import veris
from veris.grid.halos import fill_state_overlap
from veris.model.diagnostics import Diagnostics
from veris.model.types import State
from veris.physics.dynamics.driver import dynamics_transport
from veris.physics.thermodynamics.growth import Growth
from veris.setups import island


@pytest.mark.parametrize("compiled", [False, True])
def test_unequal_kernel_timesteps_match_frozen_original(compiled: bool) -> None:
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

    def compose(initial: State) -> tuple[State, Diagnostics]:
        """Apply original forcing and stage order with independent intervals."""
        prescribed = replace(
            initial,
            Qnet=jnp.full_like(initial.Qnet, 80.0),
            Qsw=jnp.full_like(initial.Qsw, 0.0),
        )
        transported, stress_u, stress_v = dynamics_transport(prescribed, conf, phys)
        values = Growth(transported, conf, phys)
        grown = replace(
            transported,
            hIceMean=values[0],
            hSnowMean=values[1],
            Area=values[2],
            TSurf=values[3],
            Qsw=values[6],
            Qnet=values[7],
            SeaIceLoad=values[8],
            recip_hIceMean=values[10],
        )
        diagnostics = Diagnostics(values[9], stress_u, stress_v, values[4], values[5])
        return fill_state_overlap(grown, conf), fill_state_overlap(diagnostics, conf)

    numerical, diagnostics = (jax.jit(compose) if compiled else compose)(state)
    assert conf.deltatDyn == 600 and conf.deltatTherm == 3600
    fixture = (
        Path(__file__).resolve().parents[1] / "reference_data"
    ) / "island_unequal_legacy_v1.npz"
    metadata = json.loads(fixture.with_suffix(".json").read_text())
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["archive_sha256"]
    )
    assert metadata["source_commit"] == "fff4cd4e08613e8ec2359e316c94435e81682400"
    with np.load(fixture, allow_pickle=False) as original:
        assert len(original.files) == 75
        for prefix, record in (("state", numerical), ("diagnostics", diagnostics)):
            for item in fields(record):
                actual = np.asarray(getattr(record, item.name))
                expected = original[f"{prefix}.{item.name}"]
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                assert np.isfinite(actual).all()
                error = np.abs(actual - expected)
                assert np.all(error <= 2e-11 * (1 + np.abs(expected))), (
                    f"ERROR compiled={compiled} {prefix}.{item.name}: max_abs={error.max():.6g}"
                )


def test_public_coupled_execution_rejects_unequal_timesteps() -> None:
    """Canonical coupled execution retains its single-clock time contract."""
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
