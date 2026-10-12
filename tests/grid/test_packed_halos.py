"""The shared packed-halo operator retains setup behavior and differentiability."""

from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh


def test_packed_halo_removal_has_grid_ownership_and_setup_compatibility() -> None:
    """A single shard still removes exactly two edges and its AD selects the interior."""
    operator = import_module("veris.grid.halos").remove_halos
    assert operator is import_module("veris.setups.run_parallel").remove_halos
    assert operator.__module__ == "veris.grid.halos"
    mesh = Mesh(
        np.asarray(jax.devices("cpu")[:1], dtype=object).reshape(1, 1), ("x", "y")
    )
    values = jnp.arange(42.0).reshape(6, 7)
    result = operator(values, mesh)
    np.testing.assert_array_equal(result, values[2:-2, 2:-2])
    expected = np.zeros((6, 7))
    expected[2:-2, 2:-2] = 1
    gradient = jax.jit(jax.grad(lambda value: operator(value, mesh).sum()))(values)
    np.testing.assert_array_equal(gradient, expected)
