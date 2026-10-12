"""Branch-safe primitives for the sea-ice model's piecewise differentiable maps.

A Euclidean norm has no unique classical derivative at its origin. The model
selects a zero linearization there, matching symmetric coordinate differences.
Positive squared norms retain their original values and derivatives; this does
not introduce smoothing or change the forward reference equations. Guarding
before sqrt prevents an inactive branch's infinite derivative contaminating AD.
"""

import jax
import jax.numpy as jnp
from jax import Array

from veris.model.types import ArrayInput


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


@jax.custom_jvp
def norm_hypot(u: ArrayInput, v: ArrayInput) -> Array:
    """Return a stable vector magnitude with zero linearization at calm wind.

    The forward value is JAX hypot, avoiding overflow from intermediate squares.
    The analytic unit-vector JVP avoids overflow in differentiating its internal
    ratios and selects the model's zero norm convention at the origin.
    """
    return jnp.hypot(u, v)


@norm_hypot.defjvp
def _norm_hypot_jvp(
    primals: tuple[ArrayInput, ArrayInput], tangents: tuple[ArrayInput, ArrayInput]
) -> tuple[Array, Array]:
    """Differentiate a norm using bounded direction cosines, including zero."""
    u, v = (jnp.asarray(value) for value in primals)
    du, dv = tangents
    magnitude = norm_hypot(u, v)
    zero = magnitude == 0
    denominator = jnp.where(zero, jnp.ones_like(magnitude), magnitude)
    direction_u = jnp.where(zero, jnp.zeros_like(magnitude), u / denominator)
    direction_v = jnp.where(zero, jnp.zeros_like(magnitude), v / denominator)
    return magnitude, direction_u * du + direction_v * dv
