"""LANL bulk flux limits: Stefan-Boltzmann radiation and zero wind exchange."""

import importlib
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("humidity_ratio", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("temperature", [260.0, 280.0, 300.0])
@pytest.mark.parametrize("wind", [0.0, 3.0, -7.0])
def test_lanl_radiation_drag_and_saturated_equilibrium(
    sett, temperature, wind, humidity_ratio
):
    module = importlib.import_module("veris.heat_flux_MITgcm")
    values = dict(sett._asdict(), grav=sett.gravity)
    settings = namedtuple("BulkSettings", values)(**values)
    state = namedtuple("BulkState", ["settings"])(settings)
    ones = jnp.ones((3, 5))
    humidity = (
        3.797915
        * np.exp(settings.latvap * (7.93252e-6 - 2.166847e-3 / temperature))
        / 1013
    )
    result = module.bulkf_formula_lanl(
        state,
        wind * ones,
        0 * ones,
        (temperature - settings.gamma_blk * 2) * ones,
        humidity * humidity_ratio * ones,
        temperature * ones,
        ones,
    )
    radiation, latent, sensible, derivative, tau_u, tau_v, evap, qsat, devdt = result
    np.testing.assert_allclose(
        radiation, settings.ocean_emissivity * settings.stefBoltz * temperature**4
    )
    for value in (sensible, tau_v):
        np.testing.assert_allclose(value, 0, atol=1e-10)
    np.testing.assert_allclose(latent, -settings.latvap * evap, atol=1e-10)
    if humidity_ratio == 1 or wind == 0:
        np.testing.assert_allclose(latent, 0, atol=1e-10)
    else:
        assert np.all(np.asarray(latent) * (humidity_ratio - 1) > 0)
    speed = max(abs(wind), 1)
    drag = 0.0027 / speed + 0.000142 + 0.0000764 * speed
    np.testing.assert_allclose(
        tau_u, settings.rhoAir * drag * abs(wind) * wind, atol=1e-13
    )
    np.testing.assert_allclose(qsat, humidity, rtol=1e-13)
    assert np.all(np.asarray(derivative) < 0)
    assert np.all(np.asarray(devdt) >= 0)
