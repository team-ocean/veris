"""Packed halo trimming reuses tracing while preserving real array and AD behavior."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from veris.grid.halos import remove_halos


def test_halo_compilation_reuses_trace_for_dynamic_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct fields share compilation; dtype and shape changes remain distinct."""
    remove_halos.clear_cache()
    original = jax.shard_map
    traces = []

    def record_shard_map(*args: Any, **kwargs: Any) -> Any:
        traces.append(kwargs["mesh"])
        return original(*args, **kwargs)

    monkeypatch.setattr(jax, "shard_map", record_shard_map)
    mesh = Mesh(
        np.asarray(jax.devices("cpu")[:1], dtype=object).reshape(1, 1), ("x", "y")
    )
    for offset in (0.0, 7.0, -3.0):
        values = jnp.arange(42, dtype=jnp.float64).reshape(6, 7) + offset
        np.testing.assert_array_equal(remove_halos(values, mesh), values[2:-2, 2:-2])
    assert len(traces) == 1, "ERROR identical field signature retraced halo removal"
    values32 = jnp.arange(42, dtype=jnp.float32).reshape(6, 7)
    result32 = remove_halos(values32, mesh)
    assert result32.dtype == jnp.float32
    np.testing.assert_array_equal(result32, values32[2:-2, 2:-2])
    assert len(traces) == 2, "ERROR changed dtype reused incompatible trace"
    larger = jnp.arange(56, dtype=jnp.float64).reshape(7, 8)
    np.testing.assert_array_equal(remove_halos(larger, mesh), larger[2:-2, 2:-2])
    assert len(traces) == 3, "ERROR changed shape reused incompatible trace"
    direction = jnp.cos(larger)
    _, tangent = jax.jvp(
        lambda value: remove_halos(value, mesh), (larger,), (direction,)
    )
    np.testing.assert_array_equal(tangent, direction[2:-2, 2:-2])
    _, pullback = jax.vjp(lambda value: remove_halos(value, mesh), larger)
    expected = np.zeros((7, 8))
    expected[2:-2, 2:-2] = 1
    np.testing.assert_array_equal(pullback(jnp.ones((3, 4)))[0], expected)
