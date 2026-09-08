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


@pytest.mark.parametrize("ice", [1.0, 2.0])
@pytest.mark.parametrize("snow", [0.0, 0.2])
def test_surface_temperature_longwave_sensitivity(state, sett, ice, snow):
    """Differentiate the thermal iteration at a smooth, subfreezing equilibrium."""
    from veris.solve4temp import solve4temp

    temperature = 260.0
    freezing = sett.celsius2K + sett.tempFrz
    conductivity = 1 / (ice / sett.iceConduct + snow / sett.snowConduct)
    emissivity = sett.snowEmiss if snow else sett.iceEmiss
    longwave = (
        sett.stefBoltz * temperature**4
        - conductivity * (freezing - temperature) / emissivity
    )
    vapor_pressure = 10 ** (12.537 - 2663.5 / temperature)
    humidity = 0.622 * vapor_pressure / (100000 - 0.378 * vapor_pressure)
    ones = jnp.ones((3, 5))
    vs = state(
        LWdown=longwave * ones,
        SWdown=0 * ones,
        ATemp=temperature * ones,
        aqh=humidity * ones,
        wSpeed=5 * ones,
        fCori=1e-4 * ones,
    )

    def surface(radiation):
        current = vs._replace(LWdown=radiation * ones)
        return jnp.mean(
            solve4temp(
                current,
                sett,
                ice * ones,
                snow * ones,
                temperature * ones,
                freezing * ones,
            )[0]
        )

    actual = jax.grad(surface)(longwave)
    finite_difference = (surface(longwave + 1e-3) - surface(longwave - 1e-3)) / 2e-3
    humidity_slope = (
        0.622
        * 100000
        / (100000 - 0.378 * vapor_pressure) ** 2
        * vapor_pressure
        * np.log(10)
        * 2663.5
        / temperature**2
    )
    expected = emissivity / (
        conductivity
        + 4 * emissivity * sett.stefBoltz * temperature**3
        + sett.dalton * sett.cpAir * sett.rhoAir * 5
        + sett.dalton * sett.lhSublim * sett.rhoAir * 5 * humidity_slope
    )
    assert actual > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    np.testing.assert_allclose(actual, finite_difference, rtol=1e-7)
