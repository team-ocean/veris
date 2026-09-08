"""Combine already-local diagnostic totals across explicit mesh axes.

Call from inside shard_map with its mesh axis names to reduce across devices
and processes. With no axes, the local scalar or vector is returned unchanged.
Callers exclude halo duplicates before forming each local total. JAX supplies
the collective's forward and reverse differentiation rules.
"""

from typing import TypeVar

import jax
from jax import Array
from jax.typing import ArrayLike

SumInput = TypeVar("SumInput", bound=ArrayLike)


def global_sum(value: SumInput, axis_names: tuple[str, ...] = ()) -> SumInput | Array:
    """Sum local totals over named mesh axes, preserving component dimensions."""
    return jax.lax.psum(value, axis_names) if axis_names else value
