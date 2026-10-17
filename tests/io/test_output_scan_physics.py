"""Scheduled float64 output must preserve actual thermodynamic evolution."""

from dataclasses import fields
from datetime import timedelta
from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

from veris import Experiment, initialize_experiment
from veris.execution.advance import advance as advance_model
from veris.execution.rollout_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream, read_record
from veris.model.diagnostics import Diagnostics
from veris.model.types import State


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_growth_scheduled_means_preserve_plain_physics(
    tmp_path: Path, dtype: str
) -> None:
    """Compare every State leaf and a direct half-open mean on a real column."""
    physics_x64 = dtype == "float64"
    previous_x64 = jax.config.x64_enabled
    with jax.enable_x64(physics_x64):
        initial, conf, phys = initialize_experiment(
            Experiment.builtin("growth", settings={"dtype": dtype})
        )
        transition = partial(
            advance_model, configuration=conf, constants=phys, physics="thermodynamics"
        )

        def evolve(carry: tuple[State, Diagnostics]) -> tuple[State, Diagnostics]:
            return transition(carry[0])

        zeros = jax.numpy.zeros_like(initial.Area)
        carry = (initial, Diagnostics(zeros, zeros, zeros, zeros, zeros, zeros))
        interval = timedelta(seconds=float(conf.deltatTherm))
        settings = OutputSettings(
            streams=(Stream("mean", ("hIceMean", "TSurf"), interval, 2 * interval),)
        )
        plain_carry, _, _ = run_timed(carry, evolve, 2)
        path = tmp_path / "scheduled.nc"
        with OutputManager(path, settings) as manager:
            observe, select_state = output_callbacks(manager, float(conf.deltatTherm))
            assert select_state is not None

            def select(carry: tuple[State, Diagnostics]) -> object:
                return select_state(carry[0])

            scheduled_carry, _, _ = run_timed(
                carry, evolve, 2, observe=observe, select=select
            )
        plain, scheduled = plain_carry[0], scheduled_carry[0]
        assert jax.config.x64_enabled == physics_x64
        for field in fields(initial):
            actual = np.asarray(getattr(scheduled, field.name))
            expected = np.asarray(getattr(plain, field.name))
            assert actual.dtype == np.dtype(dtype), field.name
            assert np.isfinite(actual).all(), field.name
            np.testing.assert_array_equal(actual, expected, err_msg=field.name)
        assert not np.array_equal(initial.hIceMean, plain.hIceMean), (
            "ERROR thermodynamic fixture did not evolve"
        )

        # Direct samples at t=0 and t=dt define [0, 2*dt); the final state
        # belongs to the next window and must not enter the stored mean.
        direct_path = tmp_path / "direct.nc"
        with OutputManager(direct_path, settings) as direct:
            direct.sample(initial, timedelta(0))
            direct.sample(transition(initial)[0], interval)
            direct.close(2 * interval)
        actual_record = read_record(path, stream="mean")
        expected_record = read_record(direct_path, stream="mean")
        assert actual_record.count == expected_record.count == 2
        assert actual_record.bounds == expected_record.bounds
        assert actual_record.time == expected_record.time
        for name, expected in expected_record.fields.items():
            actual = actual_record.fields[name]
            assert actual.dtype == np.float64
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=2e-7 if dtype == "float32" else 2e-14,
                atol=0,
                err_msg=name,
            )
        assert jax.config.x64_enabled == physics_x64
    assert jax.config.x64_enabled == previous_x64
