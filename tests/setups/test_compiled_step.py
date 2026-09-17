"""Whole-step compilation must retain the coupled driver and its derivatives."""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("adaptive", [False, True])
def test_compiled_step_matches_evolving_python_driver(adaptive: bool) -> None:
    """Compare all fields through changing forcing on a nonuniform masked grid."""
    from veris.setups import island

    initialize, step = island.initialize, island.compiled_step
    assert step.__wrapped__ is island.step
    assert not hasattr(island.step, "lower"), "ERROR preserve the Python driver"

    assert hasattr(step, "lower"), "ERROR coupled step must expose compiled lowering"
    initial, conf, phys = initialize(6, 9)
    conf = replace(conf, useAdaptiveEVP=adaptive)
    pattern = jnp.sin(jnp.arange(initial.uWind.size).reshape(initial.uWind.shape))
    initial = replace(initial, uWind=initial.uWind + 0.3 * pattern)
    expected = actual = initial
    for cooling in (50.0, 125.0, 80.0):
        expected = step.__wrapped__(expected, conf, phys, cooling)
        actual = step(actual, conf, phys, cooling)
        for metadata in fields(initial):
            name = metadata.name
            left, right = getattr(actual, name), getattr(expected, name)
            assert np.isfinite(right).all(), f"ERROR nonfinite reference {name}"
            np.testing.assert_allclose(
                left, right, rtol=1e-11, atol=1e-11, err_msg=f"ERROR field {name}"
            )


def test_compiled_step_cooling_jvp_vjp_and_finite_difference() -> None:
    """Dynamic cooling remains differentiable through the compiled growth step."""
    from veris.setups import island

    initialize, step = island.initialize, island.compiled_step

    initial, conf, phys = initialize(5, 7)

    def compiled(cooling: jax.Array) -> jax.Array:
        return step(initial, conf, phys, cooling).hIceMean[2:-2, 2:-2].sum()

    def reference(cooling: jax.Array) -> jax.Array:
        return step.__wrapped__(initial, conf, phys, cooling).hIceMean[2:-2, 2:-2].sum()

    cooling = jnp.asarray(100.0)
    tangent = jnp.asarray(1.0)
    _, jvp = jax.jvp(compiled, (cooling,), (tangent,))
    _, expected_jvp = jax.jvp(reference, (cooling,), (tangent,))
    vjp = jax.grad(compiled)(cooling)
    expected_vjp = jax.grad(reference)(cooling)
    finite_difference = (compiled(cooling + 0.01) - compiled(cooling - 0.01)) / 0.02
    assert float(jvp) != 0
    np.testing.assert_allclose(jvp, expected_jvp, rtol=1e-12)
    np.testing.assert_allclose(vjp, expected_vjp, rtol=1e-12)
    np.testing.assert_allclose(jvp, vjp, rtol=1e-12)
    np.testing.assert_allclose(vjp, finite_difference, rtol=1e-7)
