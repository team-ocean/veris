"""Compare smooth mass sensitivities with independent central differences."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.area_mass import SeaIceMass


@pytest.mark.parametrize("snow", [False, True])
@pytest.mark.parametrize("thickness", [0.1, 0.5, 2.0])
def test_mass_gradient(state, sett, snow, thickness):
    def total(value):
        vs = state(
            hIceMean=jnp.ones((3, 5)) * (1 if snow else value),
            hSnowMean=jnp.ones((3, 5)) * (value if snow else 0.2),
        )
        return jnp.sum(SeaIceMass(vs, sett)[0])

    derivative = jax.grad(total)(thickness)
    finite_difference = (total(thickness + 1e-4) - total(thickness - 1e-4)) / 2e-4
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-10)
    assert float(derivative) == pytest.approx(
        15 * (sett.rhoSnow if snow else sett.rhoIce)
    )


@pytest.mark.parametrize("area", [0.3, 0.7, 0.95])
def test_strength_area_sensitivity_matches_constitutive_law(state, sett, area):
    """Smooth Hibler strength has dP/dA = cStar P at positive ice thickness."""
    from veris.dynamics_routines import SeaIceStrength

    def total(value):
        vs = state(
            Area=value * jnp.ones((3, 5)),
            hIceMean=1.2 * jnp.ones((3, 5)),
            iceMask=jnp.ones((3, 5)),
        )
        return jnp.sum(SeaIceStrength(vs, sett))

    derivative = jax.grad(total)(area)
    delta = 1e-5
    finite_difference = (total(area + delta) - total(area - delta)) / (2 * delta)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7)
    np.testing.assert_allclose(derivative, sett.cStar * total(area), rtol=1e-13)
