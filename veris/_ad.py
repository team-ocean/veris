"""Branch-safe primitives for the sea-ice model's piecewise differentiable maps.

A Euclidean norm has no unique classical derivative at its origin. The model
selects a zero linearization there, matching symmetric coordinate differences.
Positive squared norms retain their original values and derivatives; this does
not introduce smoothing or change the forward reference equations. Guarding
before sqrt prevents an inactive branch's infinite derivative contaminating AD.
"""

import jax.numpy as jnp
from jax import Array

from veris._typing import ArrayInput


def norm_sqrt(squared_norm: ArrayInput) -> Array:
    """Return sqrt with zero linearization at zero for squared-norm expressions.

    Use for nonnegative sums of squares or masked coefficients. Negative inputs
    remain invalid (NaN); this is not clipping or a general domain repair.
    The origin convention is not a claim of classical differentiability there.
    """
    value = jnp.asarray(squared_norm)
    zero = value == 0
    safe_value = jnp.where(zero, jnp.ones_like(value), value)
    return jnp.where(zero, jnp.zeros_like(value), jnp.sqrt(safe_value))
