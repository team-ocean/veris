"""Host lifecycle preserves legacy physics, absolute forcing and model time.

The maintained island transition supplies independent trajectory expectations.
These tests exercise the public host facade; scheduled files and restart belong
to their own test modules. No clock or diagnostic history enters numerical State.
"""

from collections.abc import Callable
from dataclasses import fields, replace
from datetime import timedelta
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import veris
from veris.model.diagnostics import Diagnostics
from veris.model.types import State
from veris.setups import island


def initialize(definition: veris.Experiment) -> Any:
    """Resolve the public facade lazily so missing implementation fails each test."""
    return veris.Simulation.initialize(definition)


def coupled_definition(steps: int = 5) -> veris.Experiment:
    """Define the established small coupled preset without internal orchestration."""
    return veris.Experiment.builtin(
        "island",
        grid=veris.Grid(4, 5),
        settings={"nEVPsteps": 2},
        forcing=veris.Forcing(constant={"Qnet": 85.0, "Qsw": 0.0}),
        steps=steps,
    )


@pytest.mark.parametrize("definition", [None, {}, "island"])
def test_invalid_initialize_definition_type_is_informative(definition: Any) -> None:
    """Catch treating an arbitrary object as an Experiment or failing on attributes."""
    with pytest.raises(TypeError, match="experiment|Experiment"):
        veris.Simulation.initialize(definition)


def assert_fields_close(
    actual: State | Diagnostics, expected: State | Diagnostics
) -> None:
    """Compare all named array leaves with concise aggregate failures."""
    assert type(actual) is type(expected)
    for item in fields(expected):
        left = np.asarray(getattr(actual, item.name))
        right = np.asarray(getattr(expected, item.name))
        assert left.shape == right.shape, f"ERROR {item.name} shape mismatch"
        assert left.dtype == right.dtype, f"ERROR {item.name} dtype mismatch"
        assert np.isfinite(left).all() and np.isfinite(right).all(), (
            f"ERROR {item.name} nonfinite output"
        )
        error = np.abs(left - right)
        assert np.all(error <= 2e-11 * (1 + np.abs(right))), (
            f"ERROR {item.name} max absolute difference {error.max():.6g}"
        )


def legacy_steps(
    initial: State,
    definition: veris.Experiment,
    count: int,
    observe: Callable[[State, Diagnostics], Any] | None = None,
) -> tuple[State, Diagnostics, list[Any]]:
    """Apply preset heat explicitly through its maintained coupled transition."""
    state = initial
    history = []
    diagnostic = None
    for _ in range(count):
        state, diagnostic = island.step_with_diagnostics(
            state,
            definition.configuration,
            definition.physical_constants,
            cooling=85.0,
        )
        if observe is not None:
            history.append(observe(state, diagnostic))
    assert diagnostic is not None
    return state, diagnostic, history


def test_public_custom_coupled_define_initialize_run_result() -> None:
    """Catch a facade requiring source edits, setup assembly or losing final diagnostics."""
    surface = np.ones((4, 5))
    surface[1, 2] = 0.0
    geometry = veris.OceanGeometry(
        maskT=surface[..., None],
        maskU=(surface * np.roll(surface, 1, axis=0))[..., None],
        maskV=(surface * np.roll(surface, 1, axis=1))[..., None],
        ht=-100.0 * surface,
        coriolis_t=np.full((4, 5), 1e-4),
        dxt=np.full(4, 8000.0),
        dxu=np.full(4, 8000.0),
        dyt=np.full(5, 8000.0),
        dyu=np.full(5, 8000.0),
        area_t=np.full((4, 5), 8000.0**2),
        area_u=np.full((4, 5), 8000.0**2),
        area_v=np.full((4, 5), 8000.0**2),
    )
    definition = veris.Experiment(
        grid=veris.Grid(4, 5),
        geometry=geometry,
        settings={"deltatDyn": 600.0, "deltatTherm": 600.0, "nEVPsteps": 2},
        initial_conditions={
            "hIceMean": 1.0,
            "hSnowMean": 0.05,
            "Area": 0.8,
            "TSurf": 260.0,
        },
        forcing=veris.Forcing(
            constant={
                "uWind": 5.0,
                "vWind": 0.0,
                "theta": 271.35,
                "ocSalt": 34.7,
                "ATemp": 260.0,
                "LWdown": 260.0,
                "aqh": 0.001,
                "Qnet": 85.0,
                "Qsw": 0.0,
            }
        ),
        steps=2,
    )
    with initialize(definition) as simulation:
        assert simulation.experiment is definition
        assert simulation.configuration == definition.configuration
        assert simulation.physical_constants == definition.physical_constants
        initial = simulation.state
        expected, diagnostic, _ = legacy_steps(initial, definition, 2)
        result = simulation.run()
        assert isinstance(result, veris.RunResult)
        assert_fields_close(result.state, expected)
        assert_fields_close(result.diagnostics, diagnostic)
        assert simulation.state is result.state
        assert result.steps == result.completed_steps == simulation.completed_steps == 2
        assert result.elapsed == simulation.elapsed == timedelta(seconds=1200)
        assert result.observations is None
        assert result.compilation_seconds >= 0 and result.integration_seconds >= 0
        assert not simulation.closed
    assert simulation.closed


def test_split_runs_match_uninterrupted_all_state_and_diagnostics() -> None:
    """Catch restarting a trajectory or losing the last auxiliary fields at run boundaries."""
    definition = coupled_definition()
    with initialize(definition) as split, initialize(definition) as whole:
        first = split.run(2)
        assert first.steps == first.completed_steps == 2
        assert first.elapsed == timedelta(seconds=1200)
        final = split.run(3)
        uninterrupted = whole.run(5)
        assert_fields_close(final.state, uninterrupted.state)
        assert_fields_close(final.diagnostics, uninterrupted.diagnostics)
        assert final.steps == 3 and final.completed_steps == 5
        assert final.elapsed == uninterrupted.elapsed == timedelta(seconds=3000)
        assert (
            first.observations
            is final.observations
            is uninterrupted.observations
            is None
        )
        # The retained diagnostics have spatial leaves, without a time axis.
        assert all(leaf.shape == (8, 9) for leaf in jax.tree.leaves(final.diagnostics))


def test_single_step_and_default_remaining_run_share_state_and_clock() -> None:
    """Catch step advancing a rollout or run(None) ignoring completed timesteps."""
    definition = coupled_definition(3)
    with initialize(definition) as simulation:
        expected_first, first_diagnostic, _ = legacy_steps(
            simulation.state, definition, 1
        )
        first = simulation.step()
        assert_fields_close(first.state, expected_first)
        assert_fields_close(first.diagnostics, first_diagnostic)
        assert first.steps == first.completed_steps == 1
        assert first.elapsed == timedelta(seconds=600)
        expected, diagnostic, _ = legacy_steps(expected_first, definition, 2)
        final = simulation.run()
        assert_fields_close(final.state, expected)
        assert_fields_close(final.diagnostics, diagnostic)
        assert final.steps == 2 and final.completed_steps == 3
        assert final.elapsed == timedelta(seconds=1800)
        done = simulation.run()
        assert done.steps == 0 and done.completed_steps == 3
        assert done.state is final.state and done.diagnostics is final.diagnostics
        assert done.elapsed == final.elapsed


def test_zero_run_preserves_initial_state_time_and_absent_diagnostics() -> None:
    """Catch a warmup advancing physics or inventing a first-step diagnostic."""
    with initialize(coupled_definition(0)) as simulation:
        initial = simulation.state
        zero = simulation.run(0)
        assert zero.state is initial is simulation.state
        assert zero.diagnostics is None and zero.observations is None
        assert zero.steps == zero.completed_steps == simulation.completed_steps == 0
        assert zero.elapsed == simulation.elapsed == timedelta(0)
        configured = simulation.run()
        assert configured.steps == 0 and configured.state is initial
        assert configured.diagnostics is None


def test_explicit_count_can_extend_the_configured_default_duration() -> None:
    """Catch treating the default duration as a hard cap on constant-forcing runs."""
    definition = coupled_definition(1)
    with initialize(definition) as simulation:
        expected, diagnostic, _ = legacy_steps(simulation.state, definition, 2)
        result = simulation.run(2)
        assert_fields_close(result.state, expected)
        assert_fields_close(result.diagnostics, diagnostic)
        assert result.steps == result.completed_steps == 2
        assert simulation.run().steps == 0


def test_forcing_exhaustion_is_atomic_and_simulation_remains_usable() -> None:
    """Catch evolving some requested steps before noticing an exhausted series."""
    definition = veris.Experiment.builtin(
        "island",
        grid=veris.Grid(4, 5),
        settings={"nEVPsteps": 2},
        forcing=veris.Forcing(series={"Qnet": [80.0, 110.0, 90.0]}),
        steps=3,
    )
    with initialize(definition) as simulation:
        first = simulation.step()
        before = simulation.state
        with pytest.raises(
            ValueError, match="forcing.*exhaust|forcing.*length|exceed.*forcing"
        ):
            simulation.run(3)
        assert simulation.state is before
        assert simulation.completed_steps == 1 and simulation.elapsed == first.elapsed
        assert not simulation.closed
        assert simulation.run(0).diagnostics is first.diagnostics
        result = simulation.run(2)
        assert result.completed_steps == 3 and result.steps == 2
        with pytest.raises(
            ValueError, match="forcing.*exhaust|forcing.*length|exceed.*forcing"
        ):
            simulation.step()
        assert simulation.state is result.state
        assert simulation.completed_steps == 3 and simulation.elapsed == result.elapsed
        assert not simulation.closed


@pytest.mark.parametrize("steps", [-1, True, 1.5, jnp.array(1)])
def test_invalid_counts_fail_without_evolution_or_closing(steps: Any) -> None:
    """Catch accepting ambiguous scan counts or mutating state during validation."""
    with initialize(coupled_definition(1)) as simulation:
        initial = simulation.state
        with pytest.raises((TypeError, ValueError), match="steps|count|integer"):
            simulation.run(steps)
        assert simulation.state is initial and simulation.completed_steps == 0
        assert simulation.elapsed == timedelta(0) and not simulation.closed
        assert simulation.step().completed_steps == 1


def test_exact_subsecond_model_time_uses_active_thermodynamic_timestep() -> None:
    """Catch dynamics-clock selection or float32 time drift across run boundaries."""
    definition = veris.Experiment.builtin(
        "growth",
        settings={"dtype": "float32", "deltatDyn": 0.5, "deltatTherm": 0.100001},
        steps=5,
    )
    with initialize(definition) as simulation:
        first = simulation.run(2)
        assert first.elapsed == timedelta(microseconds=200002)
        final = simulation.run()
        assert final.steps == 3 and final.completed_steps == 5
        assert final.elapsed == simulation.elapsed == timedelta(microseconds=500005)
        assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(final.state))


def test_elapsed_clock_overflow_is_rejected_before_physics_tracing() -> None:
    """Catch advancing a valid state before discovering accumulated timedelta overflow."""
    definition = veris.Experiment.builtin(
        "growth",
        settings={"deltatTherm": 60_000_000_000_000.0},
        steps=0,
    )
    assert definition.step_duration == timedelta(seconds=60_000_000_000_000)

    def must_not_trace(state: State, diagnostic: Diagnostics) -> jax.Array:
        raise AssertionError("ERROR clock overflow reached physics scan tracing")

    with initialize(definition) as simulation:
        initial = simulation.state
        with pytest.raises(ValueError, match="clock|elapsed|duration|timedelta"):
            simulation.run(2, observe=must_not_trace)
        assert simulation.state is initial and simulation.completed_steps == 0
        assert simulation.elapsed == timedelta(0) and not simulation.closed
        zero = simulation.run(0)
        assert zero.state is initial and zero.diagnostics is None
        assert zero.steps == zero.completed_steps == 0 and zero.elapsed == timedelta(0)


def test_spatial_wind_and_heat_series_keep_absolute_position_across_runs() -> None:
    """Catch resetting forcing slices at run boundaries or using stale wind magnitude."""
    x, y = np.indices((4, 5))
    wind = np.stack(
        [2.0 + 0.3 * x + 0.1 * y + offset for offset in (0.0, 1.2, -0.5, 0.7)]
    )
    wind_v = -1.0 + 0.2 * y
    heat = np.array([80.0, 115.0, 65.0, 95.0])
    definition = veris.Experiment.builtin(
        "island",
        grid=veris.Grid(4, 5),
        settings={"nEVPsteps": 2},
        forcing=veris.Forcing(
            constant={"vWind": wind_v}, series={"uWind": wind, "Qnet": heat}
        ),
        steps=4,
    )
    with initialize(definition) as simulation:
        expected = simulation.state
        for index in range(4):
            u_wind = jnp.asarray(np.pad(wind[index], 2, mode="wrap"))
            v_wind = jnp.asarray(np.pad(wind_v, 2, mode="wrap"))
            expected = replace(
                expected, uWind=u_wind, vWind=v_wind, wSpeed=jnp.hypot(u_wind, v_wind)
            )
            expected, diagnostic = island.step_with_diagnostics(
                expected,
                definition.configuration,
                definition.physical_constants,
                cooling=heat[index],
            )
            if index == 1:
                midpoint = simulation.run(2)
                assert_fields_close(midpoint.state, expected)
                assert_fields_close(midpoint.diagnostics, diagnostic)
        final = simulation.run(2)
        assert_fields_close(final.state, expected)
        assert_fields_close(final.diagnostics, diagnostic)
        assert final.completed_steps == 4 and final.elapsed == timedelta(seconds=2400)
        np.testing.assert_allclose(final.state.uWind[2:-2, 2:-2], wind[-1])
        np.testing.assert_allclose(
            final.state.wSpeed[2:-2, 2:-2], np.hypot(wind[-1], wind_v)
        )


def test_optional_pure_observations_include_only_updated_states_and_diagnostics() -> (
    None
):
    """Catch initial samples, discarded auxiliaries or default whole-State histories."""
    definition = coupled_definition(3)

    def observe(state: State, diagnostic: Diagnostics) -> dict[str, jax.Array]:
        return {
            "ice": state.hIceMean[2:-2, 2:-2].mean(),
            "freshwater": diagnostic.EmPmR[2:-2, 2:-2].mean(),
        }

    with initialize(definition) as simulation:
        expected, diagnostic, history = legacy_steps(
            simulation.state, definition, 3, observe
        )
        result = simulation.run(3, observe=observe)
        assert_fields_close(result.state, expected)
        assert_fields_close(result.diagnostics, diagnostic)
        assert set(result.observations) == {"ice", "freshwater"}
        for name in result.observations:
            actual = result.observations[name]
            wanted = jnp.stack([sample[name] for sample in history])
            assert actual.shape == (3,)
            np.testing.assert_allclose(actual, wanted, rtol=2e-11, atol=2e-11)


def test_invalid_observer_is_preflight_and_leaves_simulation_usable() -> None:
    """Catch treating malformed observer validation as a failed physical run."""
    with initialize(coupled_definition(1)) as simulation:
        initial = simulation.state
        with pytest.raises(TypeError, match="observe|observer|callable"):
            simulation.run(1, observe=3)
        assert simulation.state is initial and simulation.completed_steps == 0
        assert simulation.elapsed == timedelta(0) and not simulation.closed
        assert simulation.step().completed_steps == 1


def test_tracing_failure_closes_resources_without_advancing_host_clock() -> None:
    """A failing pure observer exercises real scan tracing and host failure cleanup."""
    simulation = initialize(coupled_definition(1))
    initial = simulation.state

    def fail_observation(state: State, diagnostic: Diagnostics) -> jax.Array:
        raise RuntimeError("intentional observer tracing failure")

    with pytest.raises(RuntimeError, match="intentional observer tracing failure"):
        simulation.run(1, observe=fail_observation)
    assert simulation.closed
    assert simulation.state is initial and simulation.completed_steps == 0
    assert simulation.elapsed == timedelta(0)
    simulation.close()


def test_context_closes_after_user_exception() -> None:
    """Catch leaking host resources when user code fails between completed runs."""
    simulation = initialize(coupled_definition(1))
    with pytest.raises(RuntimeError, match="user failure"), simulation:
        simulation.step()
        raise RuntimeError("user failure")
    assert simulation.closed and simulation.completed_steps == 1
    assert simulation.elapsed == timedelta(seconds=600)


def test_close_is_idempotent_and_execution_after_close_fails() -> None:
    """Catch reopening resources or advancing a closed lifecycle implicitly."""
    simulation = initialize(coupled_definition(1))
    initial = simulation.state
    simulation.close()
    simulation.close()
    assert simulation.closed and simulation.state is initial
    for operation in (simulation.step, lambda: simulation.run(0)):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    assert simulation.completed_steps == 0 and simulation.elapsed == timedelta(0)


def test_zero_observation_infers_empty_shapes_after_forcing_exhaustion() -> None:
    """Catch querying a nonexistent next forcing sample just to infer empty output."""
    definition = veris.Experiment.builtin(
        "growth", forcing=veris.Forcing(series={"Qnet": [80.0]}), steps=1
    )

    def observe(state: State, diagnostic: Diagnostics) -> dict[str, jax.Array]:
        return {"ice": state.hIceMean[2:-2, 2:-2], "stress": diagnostic.OceanStressU}

    with initialize(definition) as simulation:
        initial = simulation.state
        first_zero = simulation.run(0, observe=observe)
        assert first_zero.state is initial and first_zero.diagnostics is None
        assert first_zero.observations["ice"].shape == (0, 2, 2)
        assert first_zero.observations["stress"].shape == (0, 6, 6)
        completed = simulation.step()
        zero = simulation.run(0, observe=observe)
        assert zero.state is completed.state
        assert zero.diagnostics is completed.diagnostics
        assert zero.elapsed == completed.elapsed and zero.completed_steps == 1
        assert zero.steps == 0 and zero.observations["ice"].shape == (0, 2, 2)
        assert zero.observations["stress"].shape == (0, 6, 6)
        assert not simulation.closed


@pytest.mark.parametrize("operation", ["initialize", "run", "close"])
def test_host_lifecycle_rejects_jit_even_without_dynamic_arguments(
    operation: str,
) -> None:
    """Catch constant-only traced calls mutating a facade or selecting devices."""
    definition = coupled_definition(0)
    simulation = initialize(definition)
    initial = simulation.state

    def traced() -> jax.Array:
        if operation == "initialize":
            veris.Simulation.initialize(definition)
        elif operation == "run":
            simulation.run(0)
        else:
            simulation.close()
        return jnp.asarray(1.0)

    try:
        with pytest.raises(RuntimeError, match="outside JAX transformations"):
            jax.jit(traced)()
        assert simulation.state is initial and simulation.completed_steps == 0
        assert simulation.elapsed == timedelta(0) and not simulation.closed
    finally:
        simulation.close()


def test_host_facade_uses_initialized_cpu_backend_for_all_numerical_results() -> None:
    """Catch creating prescribed or auxiliary arrays on a different default backend."""
    definition = veris.Experiment.builtin(
        "growth",
        execution=veris.Execution(backend="cpu"),
        forcing=veris.Forcing(constant={"ATemp": 260.0}, series={"Qnet": [80.0, 90.0]}),
        steps=2,
    )
    with initialize(definition) as simulation:
        result = simulation.run(2, observe=lambda state, diagnostic: diagnostic.EmPmR)
        arrays = jax.tree.leaves(
            (result.state, result.diagnostics, result.observations)
        )
        assert all(device.platform == "cpu" for a in arrays for device in a.devices())


def test_execution_synchronization_failure_preserves_last_commit_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch committing returned arrays before pending device errors surface.

    Real physics executes; only its host synchronization boundary injects a
    failure, mirroring asynchronous device errors raised by block_until_ready.
    """
    simulation = initialize(coupled_definition(2))
    first = simulation.step()
    synchronize = jax.block_until_ready

    def fail_after_synchronization(value: Any) -> Any:
        synchronize(value)
        raise RuntimeError("intentional execution synchronization failure")

    monkeypatch.setattr(jax, "block_until_ready", fail_after_synchronization)
    with pytest.raises(RuntimeError, match="execution synchronization failure"):
        simulation.step()
    assert simulation.closed
    assert simulation.state is first.state and simulation.completed_steps == 1
    assert simulation.elapsed == first.elapsed == timedelta(seconds=600)
    simulation.close()
