"""Stable landfast-ice drag and sensitivities for one- to ninety-metre keels.

The Lemieux basal-drag soft threshold must approach a linear keel excess at
large thickness, without overflowing its exponential in either JAX precision.
Independent NumPy logaddexp and its analytic derivative define the reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.dynamics_routines import basal_drag_coeffs


def basal_case(state, dtype, area, thickness):
    """Retype fixture PyTree leaves explicitly to exercise actual float32 kernels."""
    ones = np.ones((3, 5))
    vs = state(
        hIceMean=thickness * ones,
        Area=area * ones,
        R_low=-8 * ones,
        maskInU=ones,
        maskInV=ones,
    )
    return jax.tree.map(lambda value: jnp.asarray(value, dtype=dtype), vs)


def stable_reference(sett, thickness, area, u, v):
    """Return coefficient and uniform-thickness/velocity derivatives in float64."""
    speed_squared = 0.5 * (u**2 + v**2) + sett.basalDragU0**2
    scale = sett.basalDragK2 / np.sqrt(speed_squared)
    scale *= np.exp(-sett.cBasalStar * (1 - area))
    argument = 10 * (thickness - 8 * area / sett.basalDragK1)
    coefficient = scale * np.logaddexp(0, argument) / 10
    thickness_derivative = scale * np.exp(-np.logaddexp(0, -argument))
    velocity_derivative = -coefficient * 0.5 * u / speed_squared
    return coefficient, thickness_derivative, velocity_derivative


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("area", [0.0, 0.01, 0.0101, 0.5, 1.0])
@pytest.mark.parametrize("thickness", [1.0, 10.0, 90.0])
def test_basal_drag_finite_and_matches_stable_keel_law(
    state, sett, dtype, area, thickness
):
    sett = sett._replace(basalDragK2=0.7)
    vs = basal_case(state, dtype, area, thickness)
    u = jnp.full(vs.Area.shape, 0.03, dtype=dtype)
    v = jnp.full(vs.Area.shape, 0.04, dtype=dtype)
    coefficient = basal_drag_coeffs(vs, sett, u, v)
    expected = (
        stable_reference(
            sett, thickness, float(dtype(area)), float(dtype(0.03)), float(dtype(0.04))
        )[0]
        if area > 0.01
        else 0
    )
    assert coefficient.dtype == dtype
    assert np.all(np.isfinite(coefficient)), (
        "ERROR basal drag overflowed for finite keel"
    )
    np.testing.assert_allclose(coefficient, expected, rtol=3e-6, atol=1e-15)
    assert np.all(np.asarray(coefficient) >= 0), "ERROR basal drag must oppose motion"


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("area", [0.0, 0.01, 0.0101, 0.5, 1.0])
@pytest.mark.parametrize("thickness", [1.0, 10.0, 90.0])
def test_basal_drag_gradients_are_finite_and_match_analytic_law(
    state, sett, dtype, area, thickness
):
    sett = sett._replace(basalDragK2=0.7)
    vs = basal_case(state, dtype, area, thickness)

    def mean_drag(height, velocity):
        current = vs._replace(hIceMean=jnp.full_like(vs.hIceMean, height))
        return jnp.mean(
            basal_drag_coeffs(
                current,
                sett,
                jnp.full_like(vs.Area, velocity),
                jnp.full_like(vs.Area, 0.04),
            )
        )

    derivatives = jax.grad(mean_drag, argnums=(0, 1))(
        jnp.asarray(thickness, dtype=dtype), jnp.asarray(0.03, dtype=dtype)
    )
    expected = (
        stable_reference(
            sett, thickness, float(dtype(area)), float(dtype(0.03)), float(dtype(0.04))
        )[1:]
        if area > 0.01
        else (0, 0)
    )
    for actual, reference in zip(derivatives, expected):
        assert actual.dtype == dtype
        assert np.isfinite(actual), "ERROR nonfinite basal drag sensitivity"
        np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=1e-15)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("thickness", [1.0, 10.0, 90.0])
def test_disabled_basal_drag_is_zero_with_zero_thickness_sensitivity(
    state, sett, dtype, thickness
):
    sett = sett._replace(basalDragK2=0)
    vs = basal_case(state, dtype, 1, thickness)
    velocity = jnp.full_like(vs.Area, 0.03)

    def mean_drag(height):
        current = vs._replace(hIceMean=jnp.full_like(vs.hIceMean, height))
        return jnp.mean(basal_drag_coeffs(current, sett, velocity, velocity))

    value, derivative = jax.value_and_grad(mean_drag)(
        jnp.asarray(thickness, dtype=dtype)
    )
    np.testing.assert_array_equal(value, 0)
    np.testing.assert_array_equal(derivative, 0)
