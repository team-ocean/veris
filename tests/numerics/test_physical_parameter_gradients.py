"""Bottom-up physical-parameter AD against independent finite differences."""

from collections.abc import Callable
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris import Experiment, Grid, initialize_experiment
from veris.execution.advance import advance
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.numerics.freedrift import freedrift_solver
from veris.numerics.momentum import tauXY


@pytest.fixture
def cold_column() -> tuple[State, Configuration, PhysicalConstants]:
    definition = Experiment.builtin(
        "growth",
        grid=Grid(2, 2),
        settings={"deltatTherm": 300.0, "deltatDyn": 300.0},
        initial_conditions={
            "Area": 0.5,
            "hIceMean": 0.25,
            "hSnowMean": 0.0,
            "TSurf": 260.0,
        },
    )
    state, conf, p = initialize_experiment(definition)
    state = replace(state, fCori=jnp.full_like(state.fCori, 1e-4))
    return state, conf, p


def assert_parameter_derivatives(
    objective: Callable[[jax.Array], jax.Array],
    values: jax.Array,
) -> None:
    """Require identifiable columns and two stable central-difference scales."""
    forward = np.asarray(jax.jacfwd(objective)(values))
    reverse = np.asarray(jax.jacrev(objective)(values))
    assert np.isfinite(forward).all() and np.isfinite(reverse).all()
    for i in range(values.size):
        scale = np.linalg.norm(forward[:, i])
        assert scale > 1e-12, f"ERROR insensitive chosen parameter {i}"
        assert np.linalg.norm(forward[:, i] - reverse[:, i]) / scale < 1e-10
        errors = []
        for relative in (1e-2, 1e-3, 1e-4):
            step = float(values[i]) * relative
            direction = jnp.zeros_like(values).at[i].set(step)
            difference = np.asarray(
                (objective(values + direction) - objective(values - direction))
                / (2 * step)
            )
            errors.append(np.linalg.norm(difference - forward[:, i]) / scale)
        assert max(errors[1:]) <= 2e-4, (
            f"ERROR finite difference parameter {i}: {errors}"
        )
    tangent = jnp.arange(1.0, values.size + 1)
    value, jvp = jax.jvp(objective, (values,), (tangent,))
    cotangent = jnp.arange(1.0, value.size + 1)
    _, pullback = jax.vjp(objective, values)
    np.testing.assert_allclose(
        jnp.vdot(jvp, cotangent), jnp.vdot(tangent, pullback(cotangent)[0]), rtol=1e-11
    )


def test_thermodynamic_parameter_jvp_vjp_and_finite_difference_plateau(
    cold_column: tuple[State, Configuration, PhysicalConstants],
) -> None:
    state, conf, p = cold_column
    forcing = {
        "Qnet": 100.0,
        "Qsw": 20.0,
        "ocSalt": 35.0,
        "theta": 271.15,
        "ATemp": 260.0,
        "LWdown": 180.0,
        "SWdown": 30.0,
        "aqh": 0.001,
        "wSpeed": 8.0,
        "snowfall": 0.0,
        "precip": 0.0,
        "evap": 0.0,
        "runoff": 0.0,
    }

    @jax.jit
    def objective(parameters: jax.Array) -> jax.Array:
        current = p.with_overrides(
            h0=parameters[0], iceConduct=parameters[1], dryIceAlb=parameters[2]
        )
        result, _diagnostics = advance(
            state, conf, current, forcing, physics="thermodynamics"
        )
        return jnp.stack(
            [
                jnp.mean(field[2:-2, 2:-2])
                for field in (result.Area, result.hIceMean, result.TSurf)
            ]
        )

    assert_parameter_derivatives(
        objective, jnp.asarray([0.5, p.iceConduct, p.dryIceAlb])
    )


def test_air_and_water_drag_parameter_jvp_vjp_and_finite_difference_plateau(
    cold_column: tuple[State, Configuration, PhysicalConstants],
) -> None:
    state, conf, p = cold_column
    state = replace(
        state,
        uWind=jnp.full_like(state.uWind, 6.0),
        vWind=jnp.full_like(state.vWind, 2.0),
    )

    @jax.jit
    def objective(parameters: jax.Array) -> jax.Array:
        current = p.with_overrides(airIceDrag=parameters[0], waterIceDrag=parameters[1])
        tx, ty = tauXY(state, conf, current)
        velocity = freedrift_solver(
            replace(state, WindForcingX=0.5 * tx, WindForcingY=0.5 * ty), conf, current
        )
        return jnp.stack([jnp.mean(component[2:-2, 2:-2]) for component in velocity])

    assert_parameter_derivatives(objective, jnp.asarray([p.airIceDrag, p.waterIceDrag]))


def test_hard_cleanup_cutoff_keeps_branch_local_and_one_sided_behavior(
    cold_column: tuple[State, Configuration, PhysicalConstants],
) -> None:
    """The discontinuous cutoff has a finite selected derivative, without smoothing."""
    from veris.physics.transport.cleanup import clean_up_advection

    state, conf, p = cold_column

    @jax.jit
    def retained(threshold: jax.Array) -> jax.Array:
        physical = p.with_overrides(hIce_min=threshold)
        return jnp.mean(clean_up_advection(state, conf, physical)[0])

    left, at, right = 0.25 - 1e-6, 0.25, 0.25 + 1e-6
    np.testing.assert_allclose(
        [retained(jnp.asarray(x)) for x in (left, at, right)], [0.25, 0.0, 0.0]
    )
    for value in (left, at, right):
        np.testing.assert_array_equal(jax.grad(retained)(jnp.asarray(value)), 0.0)
