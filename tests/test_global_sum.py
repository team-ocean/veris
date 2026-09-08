"""Serial local totals and real multi-process global reduction semantics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFieldInput

from veris.global_sum import global_sum


@pytest.mark.parametrize("value", [2.5, [3.0, 7.0], [[1.0, 2.0], [3.0, 4.0]]])
def test_serial_global_sum_preserves_already_local_totals(
    value: StateFieldInput,
) -> None:
    """Serial reduction preserves components; it must not sum array axes again."""
    value = jnp.asarray(value)
    result = global_sum(value, axis_names=())
    np.testing.assert_array_equal(result, value)
    np.testing.assert_array_equal(jax.grad(lambda x: jnp.sum(global_sum(x)))(value), 1)


def test_two_process_cpu_reductions_and_gradients() -> None:
    """Cross-process collectives must include unequal ranks and exclude halos."""
    from reduction_probe import launch_processes

    outputs = launch_processes(platform="cpu", count=2, devices_per_process=1)
    for rank, output in enumerate(outputs):
        assert (
            f"rank {rank}: reduction values, gradients, and halo exclusion passed"
            in output
        )
