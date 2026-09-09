"""Compare smooth mass sensitivities with independent central differences."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFactory
from jax import Array
from jax.typing import ArrayLike

from veris.area_mass import SeaIceMass
from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants


@pytest.mark.parametrize("snow", [False, True])
@pytest.mark.parametrize("thickness", [0.1, 0.5, 2.0])
def test_mass_gradient(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    snow: bool,
    thickness: float,
) -> None:
    def total(value: ArrayLike) -> Array:
        vs = state(
            hIceMean=jnp.ones((3, 5)) * (1 if snow else value),
            hSnowMean=jnp.ones((3, 5)) * (value if snow else 0.2),
        )
        return jnp.sum(SeaIceMass(vs, sett, phys)[0])

    derivative = jax.grad(total)(thickness)
    finite_difference = (total(thickness + 1e-4) - total(thickness - 1e-4)) / 2e-4
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-10)
    assert float(derivative) == pytest.approx(
        15 * (phys.rhoSnow if snow else phys.rhoIce)
    )


@pytest.mark.parametrize("area", [0.3, 0.7, 0.95])
def test_strength_area_sensitivity_matches_constitutive_law(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, area: float
) -> None:
    """Smooth Hibler strength has dP/dA = cStar P at positive ice thickness."""
    from veris.dynamics_routines import SeaIceStrength

    def total(value: ArrayLike) -> Array:
        vs = state(
            Area=value * jnp.ones((3, 5)),
            hIceMean=1.2 * jnp.ones((3, 5)),
            iceMask=jnp.ones((3, 5)),
        )
        return jnp.sum(SeaIceStrength(vs, sett, phys))

    derivative = jax.grad(total)(area)
    delta = 1e-5
    finite_difference = (total(area + delta) - total(area - delta)) / (2 * delta)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7)
    np.testing.assert_allclose(derivative, phys.cStar * total(area), rtol=1e-13)


@pytest.mark.parametrize("ice", [1.0, 2.0])
@pytest.mark.parametrize("snow", [0.0, 0.2])
def test_surface_temperature_longwave_sensitivity(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    ice: float,
    snow: float,
) -> None:
    """Differentiate the thermal iteration at a smooth, subfreezing equilibrium."""
    from veris.solve4temp import solve4temp

    temperature = 260.0
    freezing = phys.celsius2K + phys.tempFrz
    conductivity = 1 / (ice / phys.iceConduct + snow / phys.snowConduct)
    emissivity = phys.snowEmiss if snow else phys.iceEmiss
    longwave = (
        phys.stefBoltz * temperature**4
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

    def surface(radiation: ArrayLike) -> Array:
        current = replace(vs, LWdown=radiation * ones)
        return jnp.mean(
            solve4temp(
                current,
                sett,
                phys,
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
        + 4 * emissivity * phys.stefBoltz * temperature**3
        + phys.dalton * phys.cpAir * phys.rhoAir * 5
        + phys.dalton * phys.lhSublim * phys.rhoAir * 5 * humidity_slope
    )
    assert actual > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    np.testing.assert_allclose(actual, finite_difference, rtol=1e-7)
