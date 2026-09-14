"""Free-drift AD must retain the Coriolis response at zero net forcing."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFactory

from veris.configuration import Configuration
from veris.freedrift_solver import freedrift_solver
from veris.physical_constants import PhysicalConstants


@pytest.mark.parametrize("coriolis", [-1e-4, 1e-4])
@pytest.mark.parametrize("forcing", [0.0, 1e-10, 0.05])
def test_free_drift_forcing_jvp_vjp(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    coriolis: float,
    forcing: float,
) -> None:
    """Compare both AD modes to momentum response, including its regular origin."""
    ones = jnp.ones((4, 7))

    def velocity(wind: jax.Array) -> jax.Array:
        vs = state(
            WindForcingX=wind[0] * ones,
            WindForcingY=wind[1] * ones,
            hIceMean=ones,
            fCori=coriolis * ones,
            uOcean=0 * ones,
            vOcean=0 * ones,
            iceMaskU=ones,
            iceMaskV=ones,
        )
        u, v = freedrift_solver(vs, conf, phys)
        return jnp.stack((jnp.mean(u), jnp.mean(v)))

    wind = jnp.array([forcing, -0.3 * forcing])
    tangent = jnp.array([0.7, -0.2])
    cotangent = jnp.array([0.4, 0.9])
    _, jvp = jax.jvp(velocity, (wind,), (tangent,))
    _, pullback = jax.vjp(velocity, wind)
    vjp = pullback(cotangent)[0]
    step = 1e-9 if forcing < 1e-8 else 1e-6
    basis = jnp.eye(2)
    fd = jnp.stack(
        [
            (velocity(wind + step * e) - velocity(wind - step * e)) / (2 * step)
            for e in basis
        ],
        axis=1,
    )
    np.testing.assert_allclose(jvp, fd @ tangent, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(vjp, cotangent @ fd, rtol=2e-5, atol=1e-6)
    if forcing == 0:
        mass_coriolis = phys.rhoIce * coriolis
        expected = jnp.array([[0.0, 1.0], [-1.0, 0.0]]) / mass_coriolis
        np.testing.assert_allclose(jvp, expected @ tangent, rtol=1e-12)
        np.testing.assert_allclose(vjp, cotangent @ expected, rtol=1e-12)


def test_free_drift_without_coriolis_nonzero_forcing_derivative(
    state: StateFactory, conf: Configuration, phys: PhysicalConstants
) -> None:
    """Quadratic drag has the analytic square-root response away from zero."""
    ones = jnp.ones((4, 7))

    def zonal_velocity(wind: jax.Array | float) -> jax.Array:
        vs = state(
            WindForcingX=wind * ones,
            WindForcingY=0 * ones,
            hIceMean=ones,
            fCori=0 * ones,
            uOcean=0 * ones,
            vOcean=0 * ones,
            iceMaskU=ones,
            iceMaskV=ones,
        )
        return jnp.mean(freedrift_solver(vs, conf, phys)[0])

    wind = 0.05
    drag = phys.rhoSea * phys.waterIceDrag
    expected = 0.5 / np.sqrt(drag * wind)
    _, tangent = jax.jvp(zonal_velocity, (wind,), (1.0,))
    np.testing.assert_allclose(tangent, expected, rtol=1e-12)
    np.testing.assert_allclose(jax.grad(zonal_velocity)(wind), expected, rtol=1e-12)


def test_free_drift_joint_zero_convention_and_singular_response(
    state: StateFactory, conf: Configuration, phys: PhysicalConstants
) -> None:
    """A zero AD convention must not be mistaken for a bounded true derivative."""
    ones = jnp.ones((4, 7))

    def velocity(wind: jax.Array | float) -> jax.Array:
        vs = state(
            WindForcingX=wind * ones,
            WindForcingY=0 * ones,
            hIceMean=ones,
            fCori=0 * ones,
            uOcean=0 * ones,
            vOcean=0 * ones,
            iceMaskU=ones,
            iceMaskV=ones,
        )
        return jnp.mean(freedrift_solver(vs, conf, phys)[0])

    value, tangent = jax.jvp(velocity, (0.0,), (1.0,))
    np.testing.assert_array_equal(value, 0)
    np.testing.assert_array_equal(tangent, 0)
    np.testing.assert_array_equal(jax.grad(velocity)(0.0), 0)
    epsilon = 1e-8
    quotient = velocity(epsilon) / epsilon
    smaller_quotient = velocity(epsilon / 4) / (epsilon / 4)
    # Quadratic drag gives u = sqrt(wind / drag). The right difference
    # quotient grows without bound as the step tends to zero.
    np.testing.assert_allclose(smaller_quotient, 2 * quotient, rtol=1e-12)
    np.testing.assert_allclose(
        quotient, 1 / np.sqrt(phys.rhoSea * phys.waterIceDrag * epsilon), rtol=1e-12
    )


def test_free_drift_float32_tiny_forcing_response(
    state: StateFactory, conf: Configuration, phys: PhysicalConstants
) -> None:
    """Keep weak-forcing velocities and gradients in actual single precision."""
    conf = replace(conf, dtype="float32")
    phys = replace(phys, dtype="float32")
    ones = jnp.ones((4, 7), dtype=jnp.float32)
    coriolis = 1e-4

    def velocity(wind: jax.Array) -> jax.Array:
        vs = state(
            WindForcingX=wind[0] * ones,
            WindForcingY=wind[1] * ones,
            hIceMean=ones,
            fCori=coriolis * ones,
            uOcean=0 * ones,
            vOcean=0 * ones,
            iceMaskU=ones,
            iceMaskV=ones,
        )
        # The shared state fixture uses float64 by design; explicitly cast
        # every leaf so this exercises float32 arithmetic in the solver.
        vs = jax.tree.map(lambda value: value.astype(jnp.float32), vs)
        u, v = freedrift_solver(vs, conf, phys)
        return jnp.stack((jnp.mean(u), jnp.mean(v)))

    wind = jnp.array([1e-10, -3e-11], dtype=jnp.float32)
    direction = jnp.array([0.7, -0.2], dtype=jnp.float32)
    value, tangent = jax.jvp(velocity, (wind,), (direction,))
    jacobian = np.array([[0.0, 1.0], [-1.0, 0.0]]) / (phys.rhoIce * coriolis)
    assert value.dtype == tangent.dtype == jnp.float32
    # At this forcing scale, the physical quadratic-drag correction to the
    # Coriolis-dominated linear solution is below single-precision accuracy.
    np.testing.assert_allclose(value, jacobian @ wind, rtol=2e-6, atol=0)
    np.testing.assert_allclose(tangent, jacobian @ direction, rtol=2e-6, atol=0)
    gradient = jax.grad(lambda force: jnp.sum(velocity(force)))(wind)
    np.testing.assert_allclose(gradient, np.ones(2) @ jacobian, rtol=2e-6, atol=0)
