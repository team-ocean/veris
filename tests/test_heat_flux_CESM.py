"""Atmospheric bulk flux checks from radiation and heat/water transfer laws."""

import importlib
from types import ModuleType

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants


@pytest.fixture
def cesm() -> ModuleType:
    """Import actual kernels; no fake Veros modules or numerical substitutes."""
    return importlib.import_module("veris.heat_flux_CESM")


@pytest.mark.parametrize("temperature", [250.0, 273.15, 300.0])
def test_saturated_specific_humidity_pressure_scaling(
    cesm: ModuleType, phys: PhysicalConstants, temperature: float
) -> None:
    pressure = jnp.array([80000.0, 100000.0, 120000.0])
    actual = cesm.qsat_august_eqn(phys, pressure, temperature)
    vapor_pressure = 133.322 * 10 ** (9.4051 - 2353 / temperature)
    np.testing.assert_allclose(actual * pressure, 0.622 * vapor_pressure, rtol=1e-13)
    np.testing.assert_allclose(
        cesm.qsat(phys, temperature), 640380 * np.exp(-5107.4 / temperature), rtol=1e-13
    )


@pytest.mark.parametrize("wind", [0.0, 2.0, 10.0])
@pytest.mark.parametrize("temperature", [260.0, 280.0, 300.0])
def test_simple_flux_equilibrium_and_temperature_derivative(
    cesm: ModuleType,
    sett: Settings,
    phys: PhysicalConstants,
    wind: float,
    temperature: float,
) -> None:
    s = phys
    ones = jnp.ones((3, 5))
    mask = ones.at[0, 0].set(0)
    pressure = 100000 * ones
    humidity = 0.622 / 100000 * 10 ** (9.4051 - 2353 / temperature) * 133.322 * ones

    def flux(surface: float) -> tuple[Array, Array, Array]:
        return cesm.flux_atmOcn_simple(
            sett,
            phys,
            mask,
            pressure,
            humidity,
            1.3 * ones,
            wind * ones,
            0 * ones,
            temperature * ones,
            0 * ones,
            0 * ones,
            surface * ones,
        )

    result = flux(temperature)
    np.testing.assert_allclose(
        result[0], -s.stefBoltz * temperature**4 * mask, rtol=1e-13
    )
    for value in result[1:]:
        np.testing.assert_allclose(value, 0, atol=1e-12)
    delta = 1e-3
    expected = [
        (a - b) / (2 * delta)
        for a, b in zip(flux(temperature + delta), flux(temperature - delta))
    ]
    actual = cesm.dqnetdt(
        sett,
        phys,
        mask,
        pressure,
        1.3 * ones,
        temperature * ones,
        wind * ones,
        0 * ones,
        0 * ones,
        0 * ones,
    )
    for derivative, finite_difference in zip(actual, expected):
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-8, atol=1e-10)
        assert np.all(np.asarray(derivative) <= 0)


def test_hybrid_pressure_levels(cesm: ModuleType) -> None:
    surface = jnp.array([[90000.0, 100000.0], [95000.0, 102000.0]])
    a = jnp.array([1000.0, 500.0, 0.0])
    b = jnp.array([0.0, 0.5, 1.0])
    actual = cesm.get_press_levs(surface, a, b)
    assert actual.shape == (2, 2, 3)
    for i, j, k in np.ndindex(actual.shape):
        assert float(actual[i, j, k]) == pytest.approx(
            float(a[k] + b[k] * surface[i, j])
        )


@pytest.mark.parametrize("wind", [0.5, 3.0, 12.0])
def test_drag_and_neutral_stability_functions(
    cesm: ModuleType, phys: PhysicalConstants, wind: float
) -> None:
    assert float(cesm.cdn(phys, wind)) == pytest.approx(
        0.0027 / wind + 0.000142 + 0.0000764 * wind
    )
    assert float(cesm.psixhu(1.0)) == pytest.approx(0)
    # The reference uses rounded pi/2 (1.571), leaving this known small offset.
    assert float(cesm.psimhu(phys, 1.0)) == pytest.approx(1.571 - np.pi / 2, abs=1e-14)


@pytest.mark.parametrize("humidity_ratio", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("difference", [-3.0, 0.0, 3.0])
@pytest.mark.parametrize("wind", [0.0, 5.0, -10.0])
def test_iterative_flux_heat_water_closure_and_stress_direction(
    cesm: ModuleType,
    sett: Settings,
    phys: PhysicalConstants,
    difference: float,
    wind: float,
    humidity_ratio: float,
) -> None:
    s = phys
    ones = jnp.ones((3, 5))
    mask = ones.at[0, 0].set(0)
    temperature = 280.0
    saturation = 0.98 * 640380 * np.exp(-5107.4 / temperature) / 1.3
    result = cesm.flux_atmOcn(
        sett,
        phys,
        mask,
        1.3 * ones,
        10 * ones,
        wind * ones,
        0 * ones,
        saturation * humidity_ratio * ones,
        (temperature + difference) * ones,
        (temperature + difference) * ones,
        0 * ones,
        0 * ones,
        temperature * ones,
    )
    sensible, latent, radiation, evap, taux, tauy, *_ = result
    for value in result:
        assert value.shape == (3, 5)
        assert np.all(np.isfinite(value))
    np.testing.assert_allclose(latent, s.latvap * evap, atol=1e-10)
    if humidity_ratio == 1:
        np.testing.assert_allclose(latent, 0, atol=1e-10)
    else:
        assert np.all(np.asarray(latent) * (humidity_ratio - 1) >= 0)
        assert np.any(np.abs(np.asarray(latent)) > 0)
    np.testing.assert_allclose(radiation, -s.stefBoltz * temperature**4 * mask)
    np.testing.assert_allclose(tauy, 0, atol=1e-14)
    assert np.all(np.asarray(taux) * wind >= 0)
    assert np.all(np.asarray(sensible) * difference >= 0)
    if difference == 0:
        np.testing.assert_allclose(sensible, 0, atol=1e-12)
    for value in result[:9]:
        assert float(value[0, 0]) == 0


def test_one_layer_hydrostatic_height(
    cesm: ModuleType, sett: Settings, phys: PhysicalConstants
) -> None:
    s = phys
    t = jnp.full((2, 3, 1), 280.0)
    q = jnp.full_like(t, 0.005)
    pressure = jnp.broadcast_to(jnp.array([90000.0, 100000.0]), (2, 3, 2))
    logarithm = np.log(100000 / 90000)
    alpha = 1 - 90000 / 10000 * logarithm
    geopotential = 280 * (1 + s.zvir * 0.005) * s.rdair * alpha
    height = geopotential / s.gravity
    expected = s.radius * height / (s.radius - height)
    np.testing.assert_allclose(
        cesm.compute_z_level(s, t, q, pressure), expected, rtol=1e-13
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_cloud_coefficients_at_knots_and_midpoints(
    cesm: ModuleType, sett: Settings, phys: PhysicalConstants, reverse: bool
) -> None:
    s = phys
    latitudes = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    coefficient = np.array([0.88, 0.70, 0.50, 0.70, 0.88])
    if reverse:
        latitudes, coefficient = latitudes[::-1], coefficient[::-1]
    ones = jnp.ones((2, 5))
    temperature = 280.0
    expected = (
        -s.emissivity
        * s.stefBoltz
        * temperature**4
        * (0.39 - 0.05 * np.sqrt(sett.eps2))
        * (1 - coefficient)
    )
    result = cesm.net_lw_ocn(
        sett,
        phys,
        ones,
        jnp.asarray(latitudes),
        0 * ones,
        temperature * ones,
        temperature * ones,
        ones,
    )
    np.testing.assert_allclose(result, np.broadcast_to(expected, (2, 5)), rtol=1e-12)


def test_initialized_empirical_coefficients_are_used(
    cesm: ModuleType, phys: PhysicalConstants
) -> None:
    """Distinct initialized instances control helpers without hidden defaults."""
    from dataclasses import replace

    custom = replace(
        phys,
        neutralDragInverseWind=0.004,
        cesmSaturationHumidityScale=700000.0,
        augustVaporPressureTemperature=2400.0,
        cesmUnstableMomentumOffset=1.6,
    )
    wind, temperature, pressure = 7.0, 280.0, 100000.0
    np.testing.assert_allclose(
        cesm.cdn(custom, wind), 0.004 / wind + 0.000142 + 0.0000764 * wind
    )
    np.testing.assert_allclose(
        cesm.qsat(custom, temperature), 700000 * np.exp(-5107.4 / temperature)
    )
    np.testing.assert_allclose(
        cesm.qsat_august_eqn(custom, pressure, temperature),
        0.622 / pressure * 10 ** (9.4051 - 2400 / temperature) * 133.322,
    )
    np.testing.assert_allclose(cesm.psimhu(custom, 1.0), 1.6 - np.pi / 2)


def test_initialized_cloud_table_controls_longwave(
    cesm: ModuleType, sett: Settings, phys: PhysicalConstants
) -> None:
    """Cloud interpolation consumes the selected instance's immutable table."""
    from dataclasses import replace

    custom = replace(
        phys, longwaveCloudLatitudes=(-90.0, 90.0), longwaveCloudCoefficients=(0.2, 0.6)
    )
    ones = jnp.ones((2, 3))
    actual = cesm.net_lw_ocn(
        sett,
        custom,
        ones,
        jnp.array([-90.0, 0.0, 90.0]),
        0 * ones,
        280 * ones,
        280 * ones,
        ones,
    )
    expected = (
        -phys.emissivity
        * phys.stefBoltz
        * 280**4
        * (0.39 - 0.05 * np.sqrt(sett.eps2))
        * (1 - np.array([0.2, 0.4, 0.6]))
    )
    np.testing.assert_allclose(actual, np.broadcast_to(expected, ones.shape))
