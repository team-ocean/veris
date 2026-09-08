"""LANL bulk flux limits: Stefan-Boltzmann radiation and zero wind exchange."""

import importlib
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

from veris.state import Settings


@pytest.mark.parametrize("humidity_ratio", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("temperature", [260.0, 280.0, 300.0])
@pytest.mark.parametrize("wind", [0.0, 3.0, -7.0])
def test_lanl_radiation_drag_and_saturated_equilibrium(
    sett: Settings, temperature: float, wind: float, humidity_ratio: float
) -> None:
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


@pytest.mark.parametrize("land_value", [280.0, float("nan"), 0.0])
def test_lanl_mask_preserves_ocean_and_zeros_land_and_sensitivities(
    sett: Settings, land_value: float
) -> None:
    """MITgcm's caller gates LANL evaluation on wet cells; land is inactive."""
    import jax

    module = importlib.import_module("veris.heat_flux_MITgcm")
    values = dict(sett._asdict(), grav=sett.gravity)
    settings = namedtuple("MaskedBulkSettings", values)(**values)
    state = namedtuple("MaskedBulkState", ["settings"])(settings)
    mask = jnp.array([[1, 0, 1], [0, 1, 0]])
    wet = np.asarray(mask, dtype=bool)
    uw, vw, ta, qa, tsf = (
        jnp.full(mask.shape, value) for value in (4.0, 2.0, 275.0, 0.003, 280.0)
    )
    inputs = (uw, vw, ta, qa, tsf)
    reference = module.bulkf_formula_lanl(state, *inputs, jnp.ones_like(mask))
    uw, vw, ta, qa, tsf = (jnp.where(mask, value, land_value) for value in inputs)
    masked_inputs = (uw, vw, ta, qa, tsf)
    actual = module.bulkf_formula_lanl(state, *masked_inputs, mask)
    for result, expected in zip(actual, reference):
        np.testing.assert_allclose(np.asarray(result)[wet], np.asarray(expected)[wet])
        np.testing.assert_array_equal(np.asarray(result)[~wet], 0)

    def total(
        uw: jax.Array, vw: jax.Array, ta: jax.Array, qa: jax.Array, tsf: jax.Array
    ) -> jax.Array:
        return sum(
            (
                jnp.sum(value)
                for value in module.bulkf_formula_lanl(state, uw, vw, ta, qa, tsf, mask)
            ),
            start=jnp.asarray(0.0),
        )

    gradients = jax.grad(total, argnums=(0, 1, 2, 3, 4))(*masked_inputs)
    for derivative in gradients:
        assert np.isfinite(np.asarray(derivative)).all(), (
            "ERROR nonfinite masked gradient"
        )
        np.testing.assert_array_equal(np.asarray(derivative)[~wet], 0)
