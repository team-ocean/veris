"""Periodic halos must reproduce wrapped interior values, including corners."""

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(2, 3), (4, 7), (6, 4)])
def test_periodic_halo_matches_numpy_wrap(halo, shape):
    interior = np.arange(np.prod(shape), dtype=float).reshape(shape)
    initial = np.pad(interior, 2, constant_values=-999)
    expected = np.pad(interior, 2, mode="wrap")
    for fill in (halo.fill_circular_overlap, halo.fill_overlap):
        actual = fill(jnp.asarray(initial))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(fill(actual), expected)
    u, v = halo.fill_overlap_uv(jnp.asarray(initial), jnp.asarray(-initial))
    np.testing.assert_array_equal(u, expected)
    np.testing.assert_array_equal(v, -expected)


def test_shard_map_halo_matches_serial_on_one_device(halo):
    """Execute actual collective halo code on the available single-device mesh."""
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((1, 1), ("x", "y"))
    interior = np.arange(28.0).reshape(4, 7)
    initial = jnp.asarray(np.pad(interior, 2, constant_values=-99))
    initial = jax.device_put(initial, NamedSharding(mesh, P("x", "y")))
    fill = jax.shard_map(
        halo.fill_overlap_shard, mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
    )
    np.testing.assert_array_equal(fill(initial), np.pad(interior, 2, mode="wrap"))
