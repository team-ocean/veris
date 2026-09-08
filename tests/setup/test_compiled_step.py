"""Whole-step compilation must retain the coupled driver and its derivatives."""

from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("adaptive", [False, True])
def test_compiled_step_matches_evolving_python_driver(
    halo: ModuleType, adaptive: bool
) -> None:
    """Compare all fields through changing forcing on a nonuniform masked grid."""
    from veris.setup import artificial

    assert hasattr(artificial, "compiled_step"), (
        "ERROR explicit compiled driver missing"
    )
    initialize, step = artificial.initialize, artificial.compiled_step
    assert step.__wrapped__ is artificial.step
    assert not hasattr(artificial.step, "lower"), "ERROR preserve the Python driver"

    assert hasattr(step, "lower"), "ERROR coupled step must expose compiled lowering"
    initial, sett = initialize(6, 9)
    sett = sett._replace(useAdaptiveEVP=adaptive)
    pattern = jnp.sin(jnp.arange(initial.uWind.size).reshape(initial.uWind.shape))
    initial = initial._replace(uWind=initial.uWind + 0.3 * pattern)
    expected = actual = initial
    for cooling in (50.0, 125.0, 80.0):
        expected = step.__wrapped__(expected, sett, cooling)
        actual = step(actual, sett, cooling)
        for name, left, right in zip(initial._fields, actual, expected, strict=True):
            assert np.isfinite(right).all(), f"ERROR nonfinite reference {name}"
            np.testing.assert_allclose(
                left, right, rtol=1e-11, atol=1e-11, err_msg=f"ERROR field {name}"
            )


def test_compiled_step_cooling_jvp_vjp_and_finite_difference(halo: ModuleType) -> None:
    """Dynamic cooling remains differentiable through the compiled growth step."""
    from veris.setup import artificial

    assert hasattr(artificial, "compiled_step"), (
        "ERROR explicit compiled driver missing"
    )
    initialize, step = artificial.initialize, artificial.compiled_step

    assert hasattr(step, "lower"), "ERROR coupled step must expose compiled lowering"
    initial, sett = initialize(5, 7)

    def compiled(cooling: jax.Array) -> jax.Array:
        return step(initial, sett, cooling).hIceMean[2:-2, 2:-2].sum()

    def reference(cooling: jax.Array) -> jax.Array:
        return step.__wrapped__(initial, sett, cooling).hIceMean[2:-2, 2:-2].sum()

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
