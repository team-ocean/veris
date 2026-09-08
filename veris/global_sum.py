"""Combine already-local diagnostic totals across explicit mesh axes.

Call from inside shard_map with its mesh axis names to reduce across devices
and processes. With no axes, the local scalar or vector is returned unchanged.
Callers exclude halo duplicates before forming each local total. JAX supplies
the collective's forward and reverse differentiation rules.
"""

import jax


def global_sum(value, axis_names: tuple[str, ...] = ()):
    """Sum local totals over named mesh axes, preserving component dimensions."""
    return jax.lax.psum(value, axis_names) if axis_names else value
