"""Exact norm values with a finite, explicit origin linearization."""

from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_norm_sqrt_preserves_values_and_finite_origin_ad(dtype: type) -> None:
    """Squared-norm origin chooses zero; positive inputs keep sqrt's derivative."""
    try:
        norm_sqrt = import_module("veris._ad").norm_sqrt
    except ModuleNotFoundError:
        pytest.fail("AD norm primitive is missing")
    x = jnp.asarray([0.0, 1e-12, 2.0, 4.0], dtype=dtype)
    np.testing.assert_array_equal(norm_sqrt(x), jnp.sqrt(x))
    derivative = jax.grad(lambda v: jnp.sum(norm_sqrt(v)))(x)
    np.testing.assert_allclose(
        derivative,
        jnp.asarray([0.0, 5e5, 0.5 / np.sqrt(2.0), 0.25], dtype=dtype),
        rtol=2e-6,
    )
    origin = jnp.zeros(3, dtype=dtype)
    norm = lambda v: norm_sqrt(jnp.sum(v * v))
    np.testing.assert_array_equal(jax.jacrev(norm)(origin), 0.0)
    np.testing.assert_array_equal(jax.jacfwd(norm)(origin), 0.0)
    # Invalid negative squared norms must remain visible, never silently clipped.
    assert np.isnan(norm_sqrt(jnp.asarray(-1.0, dtype=dtype)))
