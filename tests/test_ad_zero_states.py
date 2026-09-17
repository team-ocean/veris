"""AD at zero strain, calm forcing and masked cells must remain finite."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.dynamics_routines import ocean_drag_coeffs, viscosities
from veris.dynsolver import tauXY
from veris.setups.run_dyn import compiled_step, initialize


def test_zero_strain_viscosity_finite_linearization() -> None:
    """Zero is the symmetric norm linearization; primal viscosity is unchanged."""
    state, conf, phys = initialize(4, 6)
    zeros = jnp.zeros_like(state.uIce)
    state = replace(state, SeaIceStrength=jnp.ones_like(zeros) * 1000)

    def value(strain: jax.Array) -> jax.Array:
        zeta, _, _ = viscosities(state, conf, phys, strain, zeros, zeros)
        return jnp.sum(zeta)

    primal, forward = jax.jvp(value, (zeros,), (jnp.ones_like(zeros),))
    reverse = jax.grad(value)(zeros)
    for derivative in (forward, reverse):
        assert np.isfinite(derivative).all()
        np.testing.assert_array_equal(derivative, 0.0)
    np.testing.assert_allclose(primal, zeros.size * 500 / phys.deltaMin)


@pytest.mark.parametrize("kernel", ["wind", "ocean"])
def test_zero_relative_speed_reverse_ad(kernel: str) -> None:
    """Inactive sqrt branches must not poison calm wind/relative-current AD."""
    state, conf, phys = initialize(4, 6)
    zeros = jnp.zeros_like(state.uIce)
    state = replace(
        state,
        uIce=zeros,
        vIce=zeros,
        uOcean=zeros,
        vOcean=zeros,
        uWind=zeros,
        vWind=zeros,
    )

    def value(u: jax.Array) -> jax.Array:
        if kernel == "wind":
            return jnp.sum(tauXY(replace(state, uWind=u), conf, phys)[0])
        return jnp.sum(ocean_drag_coeffs(state, conf, phys, u, zeros))

    derivative = jax.grad(value)(zeros)
    assert np.isfinite(derivative).all()
    direction = jnp.ones_like(zeros)
    _, forward = jax.jvp(value, (zeros,), (direction,))
    np.testing.assert_allclose(jnp.sum(derivative), forward, rtol=1e-12, atol=1e-12)


def test_reference_initial_dynamics_wind_gradient_matches_fd() -> None:
    """Differentiate the actual stationary, coastal experiment initial state."""
    state, conf, phys = initialize(6, 8, settings_overrides={"nEVPsteps": 2})

    def objective(scale: jax.Array) -> jax.Array:
        result = compiled_step(
            replace(state, uWind=state.uWind * scale, vWind=state.vWind * scale),
            conf,
            phys,
        )
        return jnp.sum(result.uIce[2:-2, 2:-2] ** 2 + result.vIce[2:-2, 2:-2] ** 2)

    point = jnp.asarray(1.0)
    reverse = jax.grad(objective)(point)
    _, forward = jax.jvp(objective, (point,), (jnp.asarray(1.0),))
    step = 1e-3
    fd = (objective(point + step) - objective(point - step)) / (2 * step)
    assert np.isfinite(reverse) and np.isfinite(forward)
    assert abs(float(fd)) > 1e-10
    np.testing.assert_allclose(reverse, forward, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reverse, fd, rtol=2e-5, atol=1e-10)


def test_uniform_advection_thickness_sensitivity_is_finite() -> None:
    """Zero slope ratios in inactive limiter branches preserve mass sensitivity."""
    from veris.advection import Advection

    state, conf, phys = initialize(6, 8)
    state = replace(state, uIce=0.01 * state.iceMaskU, vIce=0.02 * state.iceMaskV)

    def total(thickness: jax.Array) -> jax.Array:
        ice, _, _ = Advection(
            replace(state, hIceMean=jnp.full_like(state.hIceMean, thickness)),
            conf,
            phys,
        )
        return jnp.sum(ice[2:-2, 2:-2])

    derivative = jax.grad(total)(jnp.asarray(0.3))
    # Advection masks the final x/y land lines: (6-1)*(8-1) wet cells.
    np.testing.assert_allclose(derivative, 35.0, rtol=1e-12, atol=1e-12)


def test_dry_corner_mask_linearization_is_finite() -> None:
    """A zero-count corner must not differentiate an inactive reciprocal zero."""
    from veris.averaging import c_point_to_z_point

    state, conf, phys = initialize(4, 6)
    zeros = jnp.zeros_like(state.iceMask)

    def value(mask: jax.Array) -> jax.Array:
        return jnp.sum(
            c_point_to_z_point(
                replace(state, iceMask=mask), conf, phys, jnp.ones_like(mask)
            )
        )

    np.testing.assert_array_equal(jax.grad(value)(zeros), 0.0)


def test_zero_strain_stress_retains_linear_shear_response() -> None:
    """Guarding the norm must retain viscosity times the linear shear strain."""
    from veris.dynamics_routines import stress

    state, conf, phys = initialize(4, 6)
    ones = jnp.ones_like(state.uIce)
    state = replace(state, SeaIceStrength=1000 * ones)
    zeros = jnp.zeros_like(ones)

    def shear_response(shear: jax.Array) -> jax.Array:
        zeta, eta, pressure = viscosities(state, conf, phys, zeros, zeros, shear * ones)
        return jnp.sum(
            stress(state, conf, phys, zeros, zeros, shear * ones, zeta, eta, pressure)[
                2
            ]
        )

    point = jnp.asarray(0.0)
    derivative = jax.grad(shear_response)(point)
    delta = 1e-15
    fd = (shear_response(point + delta) - shear_response(point - delta)) / (2 * delta)
    assert float(derivative) > 0
    np.testing.assert_allclose(derivative, fd, rtol=2e-6, atol=0.0)


@pytest.mark.parametrize("no_slip", [True, False])
@pytest.mark.parametrize("adaptive", [True, False])
def test_full_stationary_state_pullback_has_no_nonfinite_leaves(
    no_slip: bool, adaptive: bool
) -> None:
    """Unused output cotangents must not contaminate any input State field."""
    state, conf, phys = initialize(
        4,
        6,
        settings_overrides={
            "nEVPsteps": 2,
            "noSlip": no_slip,
            "useAdaptiveEVP": adaptive,
        },
    )
    result, pullback = jax.vjp(lambda vs: compiled_step(vs, conf, phys), state)
    cotangent = jax.tree.map(jnp.zeros_like, result)
    cotangent = replace(cotangent, uIce=jnp.ones_like(result.uIce))
    gradient = pullback(cotangent)[0]
    from dataclasses import fields

    for field in fields(gradient):
        array = getattr(gradient, field.name)
        assert np.isfinite(array).all(), f"ERROR nonfinite State pullback: {field.name}"
