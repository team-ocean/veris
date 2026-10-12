"""Independent recurrence contracts for pure scan rollout and rematerialized AD."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import veris


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 3])
def test_recurrence_and_selected_observations(steps: int, checkpoint: bool) -> None:
    """Catch incorrect iteration counts, initial samples, and scan carry updates."""
    result, history = veris.step(
        {"value": jnp.array(1.0), "constant": jnp.ones((2, 3))},
        lambda state: {**state, "value": state["value"] * 2},
        steps,
        checkpoint=checkpoint,
        observe=lambda state: {"value": state["value"]},
    )
    np.testing.assert_array_equal(history["value"], np.array([2.0, 4.0, 8.0])[:steps])
    assert set(history) == {"value"}
    assert float(result["value"]) == 2**steps
    np.testing.assert_array_equal(result["constant"], 1)


def test_time_dependent_pytree_forcing_and_auxiliary_outputs() -> None:
    """Catch ignored or reordered forcing and lost transition diagnostics."""

    def advance(state: jax.Array, forcing: dict[str, jax.Array]) -> Any:
        return state * forcing["factor"] + forcing["offset"], {"before": state}

    result, history = veris.step(
        jnp.array(1.0),
        advance,
        3,
        inputs={
            "factor": jnp.array([2.0, 3.0, 4.0]),
            "offset": jnp.array([1.0, 2.0, 3.0]),
        },
        has_aux=True,
        observe=lambda state, aux: (state, aux["before"]),
    )
    assert float(result) == 47
    np.testing.assert_array_equal(history[0], [3.0, 11.0, 47.0])
    np.testing.assert_array_equal(history[1], [1.0, 3.0, 11.0])
    assert float(veris.step(jnp.array(1.0), lambda s: (s + 2, s), 2, has_aux=True)) == 5


def test_zero_steps_forcing_and_auxiliary_shape() -> None:
    """Empty rollouts retain original carry and correctly typed empty output."""
    result, history = veris.step(
        jnp.ones((2, 3), dtype=jnp.float32),
        lambda state, force: (state + force, jnp.sum(state)),
        0,
        inputs=jnp.empty((0, 2, 3), dtype=jnp.float32),
        has_aux=True,
        observe=lambda state, aux: {"field": state, "sum": aux},
    )
    assert result.dtype == jnp.float32
    np.testing.assert_array_equal(result, 1)
    assert history["field"].shape == (0, 2, 3)
    assert history["sum"].shape == (0,)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_jitted_value_state_and_forcing_derivatives(checkpoint: bool) -> None:
    """A nonlinear recurrence has independent analytic derivatives for all inputs."""

    def objective(x: jax.Array, forcing: jax.Array) -> jax.Array:
        return veris.step(
            x, lambda s, f: s * s + f, 2, inputs=forcing, checkpoint=checkpoint
        )

    x = jnp.array(2.0)
    forcing = jnp.array([1.0, 3.0])
    compiled = jax.jit(objective)
    assert float(compiled(x, forcing)) == 28
    dx, df = jax.grad(compiled, argnums=(0, 1))(x, forcing)
    np.testing.assert_allclose(dx, 40.0)
    np.testing.assert_allclose(df, [10.0, 1.0])
    direction = (jnp.array(0.5), jnp.array([0.2, -0.3]))
    _, tangent = jax.jvp(compiled, (x, forcing), direction)
    np.testing.assert_allclose(tangent, 21.7)
    eps = 1e-5
    fd = (
        compiled(x + eps * direction[0], forcing + eps * direction[1])
        - compiled(x - eps * direction[0], forcing - eps * direction[1])
    ) / (2 * eps)
    np.testing.assert_allclose(tangent, fd, rtol=1e-9)


def test_observation_and_closed_over_forcing_gradients() -> None:
    """Checkpoint must preserve gradients of selected histories and captured tracers."""

    def objective(force: jax.Array) -> jax.Array:
        _, history = veris.step(
            jnp.array(1.0), lambda s: s * force, 3, observe=lambda s: s
        )
        return history.sum()

    np.testing.assert_allclose(jax.jit(jax.grad(objective))(jnp.array(2.0)), 17.0)


def _primitive_names(program: Any) -> list[str]:
    """Walk nested compiled programs to inspect the actual integration strategy."""
    if hasattr(program, "eqns"):
        return [
            name
            for equation in program.eqns
            for name in [equation.primitive.name, *_primitive_names(equation.params)]
        ]
    if hasattr(program, "jaxpr"):
        return _primitive_names(program.jaxpr)
    if isinstance(program, dict):
        return [name for value in program.values() for name in _primitive_names(value)]
    if isinstance(program, (tuple, list)):
        return [name for value in program for name in _primitive_names(value)]
    return []


@pytest.mark.parametrize("checkpoint", [False, True])
def test_rollout_uses_scan_and_requested_checkpoint(checkpoint: bool) -> None:
    """Prevent replacing required scan/rematerialization with a Python unroll."""
    program = jax.make_jaxpr(
        lambda x: veris.step(x, lambda s: s * s + 0.1, 3, checkpoint=checkpoint)
    )(jnp.array(0.5))
    names = _primitive_names(program)
    assert "scan" in names
    assert any("remat" in name for name in names) == checkpoint


@pytest.mark.parametrize("steps", [-1, 1.5, True, jnp.array(2)])
def test_invalid_static_counts(steps: Any) -> None:
    """Host wrappers reject counts that cannot define a static scan length."""
    with pytest.raises((TypeError, ValueError), match="steps"):
        veris.step(jnp.array(1.0), lambda s: s, steps)


@pytest.mark.parametrize(
    "inputs",
    [jnp.array(1.0), jnp.ones((2,)), {"a": jnp.ones((3,)), "b": jnp.ones((2,))}, {}],
)
def test_invalid_forcing_lengths(inputs: Any) -> None:
    """Do not silently drop forcing slices or accept ambiguous empty PyTrees."""
    with pytest.raises(ValueError, match="inputs"):
        veris.step(jnp.array(1.0), lambda s, f: s, 3, inputs=inputs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"advance": None}, "advance"),
        ({"observe": 2}, "observe"),
        ({"checkpoint": 1}, "checkpoint"),
        ({"has_aux": "yes"}, "has_aux"),
    ],
)
def test_invalid_callables_and_static_switches(
    kwargs: dict[str, Any], message: str
) -> None:
    """Reject invalid API choices before tracing a transition."""
    options: dict[str, Any] = {"advance": lambda s: s, **kwargs}
    with pytest.raises(TypeError, match=message):
        veris.step(jnp.array(1.0), steps=0, **options)


def test_shape_changing_transition_fails_informatively() -> None:
    """A scan cannot silently resize the State partway through integration."""
    with pytest.raises(TypeError, match="(carry|shape)"):
        veris.step(jnp.ones(2), lambda s: jnp.concatenate((s, s)), 2)
