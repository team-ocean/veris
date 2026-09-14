"""Finite branch linearizations at ice-free, freshwater, and calm-wind limits.

Ice-free flux derivatives vanish, while the input surface temperature survives.
Bulk wind norms use a zero linearization at their nondifferentiable origin;
centered differences check that convention and the differentiable stress maps.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFactory
from jax import Array
from test_growth import equilibrium_state

from veris.configuration import Configuration
from veris.growth import Growth
from veris.heat_flux_CESM import dqnetdt, flux_atmOcn, flux_atmOcn_simple, net_lw_ocn
from veris.heat_flux_MITgcm import bulkf_formula_lanl
from veris.physical_constants import PhysicalConstants
from veris.solve4temp import solve4temp


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("temperature", [0.0, 265.0])
def test_absent_ice_surface_fluxes_have_finite_branch_derivatives(
    state: StateFactory, dtype: str, temperature: float
) -> None:
    """Inactive conductivity and Newton divisions must not poison pullbacks."""
    conf = Configuration(use_sharding=False, dtype=dtype)
    phys = PhysicalConstants(dtype=dtype)
    vs = state(LWdown=300, SWdown=200, ATemp=270, aqh=0.003, wSpeed=0, fCori=1e-4)
    vs = jax.tree.map(lambda x: x.astype(dtype), vs)

    def surface(inputs: Array) -> Array:
        return jnp.stack(
            solve4temp(
                vs,
                conf,
                phys,
                inputs[0],
                inputs[1],
                inputs[2],
                jnp.asarray(271, dtype=dtype),
            )
        )

    inputs = jnp.asarray([0, 0, temperature], dtype=dtype)
    expected = np.zeros((5, 3))
    expected[0, 2] = 1
    np.testing.assert_array_equal(surface(inputs), [temperature, 0, 0, 0, 0])
    np.testing.assert_array_equal(jax.jacfwd(surface)(inputs), expected)
    np.testing.assert_array_equal(jax.jacrev(surface)(inputs), expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_growth_freshwater_salinity_has_finite_one_sided_sensitivity(
    state: StateFactory, dtype: str
) -> None:
    """Inactive sea-ice salt rejection cannot divide by freshwater salinity."""
    conf = Configuration(use_sharding=False, dtype=dtype, nITC=1)
    phys = PhysicalConstants(dtype=dtype)
    vs = equilibrium_state(state, conf, phys, ocSalt=0, theta=275, TSurf=260, Qnet=100)
    vs = jax.tree.map(lambda x: x.astype(dtype), vs)

    def freshwater(salt: Array) -> Array:
        return jnp.sum(
            Growth(replace(vs, ocSalt=jnp.full_like(vs.ocSalt, salt)), conf, phys)[4]
        )

    salt = jnp.asarray(0.0, dtype=dtype)
    _, forward = jax.jvp(freshwater, (salt,), (jnp.ones_like(salt),))
    reverse = jax.grad(freshwater)(salt)
    step = 0.01 if dtype == "float32" else 1e-4
    finite_difference = (freshwater(salt + step) - freshwater(salt)) / step
    assert np.isfinite(reverse), "ERROR freshwater salinity pullback is nonfinite"
    np.testing.assert_allclose(reverse, forward, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(reverse, finite_difference, rtol=0.01, atol=1e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "kernel", ["cesm_simple", "cesm_correction", "cesm_bulk", "lanl"]
)
def test_calm_wind_fluxes_have_finite_selected_linearizations(
    dtype: str, kernel: str
) -> None:
    """An inactive speed floor must not leave sqrt(0) in forward/reverse AD."""
    conf = Configuration(use_sharding=False, dtype=dtype)
    phys = PhysicalConstants(dtype=dtype)
    one = jnp.ones((1, 1), dtype=dtype)
    zero = jnp.zeros_like(one)

    def flux(wind: Array) -> Array:
        u = one * wind
        if kernel == "cesm_simple":
            result = flux_atmOcn_simple(
                conf,
                phys,
                one,
                one * 1e5,
                one * 0.003,
                one * 1.3,
                u,
                zero,
                one * 270,
                zero,
                zero,
                one * 275,
            )
        elif kernel == "cesm_correction":
            result = dqnetdt(
                conf, phys, one, one * 1e5, one * 1.3, one * 275, u, zero, zero, zero
            )
        elif kernel == "cesm_bulk":
            result = flux_atmOcn(
                conf,
                phys,
                one,
                one * 1.3,
                one * 10,
                u,
                zero,
                one * 0.003,
                one * 270,
                one * 270,
                zero,
                zero,
                one * 275,
            )
        else:
            result = bulkf_formula_lanl(
                conf, phys, u, zero, one * 270, one * 0.003, one * 275, one
            )
        return jnp.concatenate([jnp.ravel(x) for x in result])

    wind = jnp.asarray(0, dtype=dtype)
    primal, forward = jax.jvp(flux, (wind,), (jnp.ones_like(wind),))
    reverse = jax.jacrev(flux)(wind)
    step = 1e-4 if dtype == "float32" else 1e-6
    finite_difference = (flux(wind + step) - flux(wind - step)) / (2 * step)
    assert np.isfinite(primal).all(), "ERROR calm wind flux is nonfinite"
    assert np.isfinite(forward).all(), "ERROR calm wind tangent is nonfinite"
    assert np.isfinite(reverse).all(), "ERROR calm wind pullback is nonfinite"
    np.testing.assert_allclose(reverse, forward, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(reverse, finite_difference, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("land_value", [0.0, float("nan")])
@pytest.mark.parametrize("kernel", ["simple", "correction", "bulk", "longwave"])
def test_cesm_land_inputs_are_inactive_in_fluxes_and_ad(
    kernel: str, land_value: float
) -> None:
    """Zero/undefined land forcing must not contaminate masked flux derivatives."""
    conf = Configuration(use_sharding=False)
    phys = PhysicalConstants()
    mask = jnp.asarray([[1.0, 0.0]])
    # Pressure, density, measurement height, wind, humidity, temperature, SST.
    reference = jnp.asarray([1e5, 1.3, 10, 4, 0.003, 270, 275])[:, None, None]
    values = jnp.broadcast_to(reference, (7, 1, 2)).at[:, :, 1].set(land_value)

    def flux(inputs: Array) -> Array:
        ps, rho, height, wind, humidity, temperature, sst = inputs
        zero = jnp.zeros_like(mask)
        if kernel == "simple":
            outputs = flux_atmOcn_simple(
                conf,
                phys,
                mask,
                ps,
                humidity,
                rho,
                wind,
                zero,
                temperature,
                zero,
                zero,
                sst,
            )
        elif kernel == "correction":
            outputs = dqnetdt(conf, phys, mask, ps, rho, sst, wind, zero, zero, zero)
        elif kernel == "bulk":
            outputs = flux_atmOcn(
                conf,
                phys,
                mask,
                rho,
                height,
                wind,
                zero,
                humidity,
                temperature,
                temperature,
                zero,
                zero,
                sst,
            )
        else:
            outputs = (
                net_lw_ocn(
                    conf,
                    phys,
                    mask,
                    jnp.asarray([0.0, 45.0]),
                    humidity,
                    sst,
                    temperature,
                    jnp.ones_like(mask) * 0.5,
                ),
            )
        return jnp.stack(outputs)

    expected = flux(jnp.broadcast_to(reference, (7, 1, 2)))
    actual = flux(values)
    assert np.isfinite(actual).all(), "ERROR inactive CESM land produced nonfinite flux"
    np.testing.assert_array_equal(actual[:, :, 0], expected[:, :, 0])
    np.testing.assert_array_equal(actual[:, :, 1], 0)
    tangent = jnp.zeros_like(values).at[:, :, 1].set(1)
    _, forward = jax.jvp(flux, (values,), (tangent,))
    reverse = jax.grad(lambda x: jnp.sum(flux(x)))(values)
    np.testing.assert_array_equal(forward, 0)
    assert np.isfinite(reverse).all(), "ERROR inactive CESM land poisoned pullback"
    np.testing.assert_array_equal(reverse[:, :, 1], 0)


def test_growth_zero_area_regularization_keeps_open_water_growth_finite(
    state: StateFactory,
) -> None:
    """The supported Area_reg=0 setting must permit the open-water branch."""
    conf = Configuration(use_sharding=False, nITC=1, deltatTherm=600)
    phys = PhysicalConstants(Area_reg=0)
    vs = equilibrium_state(state, conf, phys, Area=0, hIceMean=0, Qnet=100)

    def ice(cooling: Array) -> Array:
        return Growth(replace(vs, Qnet=jnp.full_like(vs.Qnet, cooling)), conf, phys)[0]

    cooling = jnp.asarray(100.0)
    expected_slope = conf.deltatTherm / (phys.rhoIce * phys.lhFusion)
    np.testing.assert_allclose(ice(cooling), 100 * expected_slope, rtol=1e-12)
    _, forward = jax.jvp(ice, (cooling,), (jnp.ones_like(cooling),))
    reverse = jax.jacrev(ice)(cooling)
    np.testing.assert_allclose(forward, expected_slope, rtol=1e-12)
    np.testing.assert_allclose(reverse, expected_slope, rtol=1e-12)


def test_lanl_stable_branch_ignores_singular_unstable_auxiliary() -> None:
    """At z/L=1/16, the inactive sqrt is zero but stable fluxes are smooth."""
    conf = Configuration(use_sharding=False, lanlBulkIterations=1)
    # rdn=1 and zero humidity give z/L=(Ta-Ts)/Ta=16/256 exactly.
    phys = PhysicalConstants(
        karman=1,
        gravity=1,
        zref=1,
        ztref=1,
        zzsice=np.exp(-1),
        gamma_blk=0,
        lanlSaturationHumidityScale=0,
    )
    one = jnp.ones((1, 1))

    def sensible(temperature: Array) -> Array:
        return bulkf_formula_lanl(
            conf, phys, one, one * 0, one * temperature, one * 0, one * 240, one
        )[2].sum()

    temperature = jnp.asarray(256.0)
    _, forward = jax.jvp(sensible, (temperature,), (jnp.ones_like(temperature),))
    reverse = jax.grad(sensible)(temperature)
    step = 1e-4
    finite_difference = (
        sensible(temperature + step) - sensible(temperature - step)
    ) / (2 * step)
    assert np.isfinite(reverse), "ERROR stable flux differentiated inactive sqrt(0)"
    np.testing.assert_allclose(forward, reverse, rtol=1e-12)
    np.testing.assert_allclose(reverse, finite_difference, rtol=1e-7)


def test_cesm_bulk_preserves_fractional_wet_mask_weights() -> None:
    """Land guards must retain the source's per-output fractional mask powers."""
    conf = Configuration(use_sharding=False)
    phys = PhysicalConstants()
    one = jnp.ones((1, 1))
    humidity = 0.003

    def flux(mask: Array) -> Array:
        return jnp.stack(
            flux_atmOcn(
                conf,
                phys,
                mask,
                one * 1.3,
                one * 10,
                one * 4,
                one * 2,
                one * humidity,
                one * 270,
                one * 270,
                one * 0,
                one * 0,
                one * 275,
            )
        )

    full = np.asarray(flux(one))
    fraction = 0.25
    expected = full * fraction
    # Evaporation applies mask twice, whereas the last three turbulent
    # diagnostics are defined per wet cell without a fractional-area weight.
    expected[3] = full[3] * fraction**2
    expected[7] = humidity * fraction - (humidity - full[7]) * fraction**2
    expected[9:] = full[9:]
    np.testing.assert_allclose(flux(one * fraction), expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(flux(one * 0), 0)


@pytest.mark.parametrize(
    "field_index",
    [4, 5, 6, 7, 8],
    ids=["longwave", "shortwave", "air_temperature", "humidity", "wind"],
)
def test_absent_ice_ignores_undefined_atmospheric_forcing(
    state: StateFactory, field_index: int
) -> None:
    """Undefined ice-free forcing must not contaminate any input pullback."""
    conf = Configuration(use_sharding=False)
    phys = PhysicalConstants()
    # Ice, snow, surface/freezing temperature, then five atmospheric forcings.
    reference = jnp.asarray([1, 0.1, 265, 271, 300, 200, 270, 0.003, 3])[:, None, None]
    inputs = jnp.broadcast_to(reference, (9, 1, 2)).at[:2, :, 1].set(0)
    vs = state(
        fCori=jnp.ones((1, 2)) * 1e-4,
        LWdown=inputs[4],
        SWdown=inputs[5],
        ATemp=inputs[6],
        aqh=inputs[7],
        wSpeed=inputs[8],
    )

    def surface(values: Array) -> Array:
        forcing = replace(
            vs,
            LWdown=values[4],
            SWdown=values[5],
            ATemp=values[6],
            aqh=values[7],
            wSpeed=values[8],
        )
        return jnp.stack(
            solve4temp(forcing, conf, phys, values[0], values[1], values[2], values[3])
        )

    expected = surface(inputs)
    inputs = inputs.at[field_index, :, 1].set(jnp.nan)
    np.testing.assert_array_equal(surface(inputs), expected)
    tangent = jnp.zeros_like(inputs).at[:, :, 1].set(1)
    _, forward = jax.jvp(surface, (inputs,), (tangent,))
    expected_forward = np.zeros((5, 1, 2))
    expected_forward[0, 0, 1] = 1
    np.testing.assert_array_equal(forward, expected_forward)
    reverse = jax.grad(lambda values: surface(values).sum())(inputs)
    assert np.isfinite(reverse).all(), (
        "ERROR undefined ice-free forcing poisoned pullback"
    )
    np.testing.assert_array_equal(reverse[:, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0])


def test_cesm_longwave_ignores_undefined_latitude_on_dry_columns() -> None:
    """Latitude is shared down columns; only entirely dry columns are inactive."""
    conf = Configuration(use_sharding=False)
    phys = PhysicalConstants()
    mask = jnp.asarray([[1, 0, 0], [0, 0, 1]])
    one = jnp.ones(mask.shape)
    reference_latitude = jnp.asarray([0.0, 20.0, 45.0])
    latitude = reference_latitude.at[1].set(jnp.nan)
    sst = one * 275

    def flux(lat: Array, temperature: Array) -> Array:
        return net_lw_ocn(
            conf, phys, mask, lat, one * 0.003, temperature, one * 270, one * 0.5
        )

    np.testing.assert_array_equal(flux(latitude, sst), flux(reference_latitude, sst))
    _, forward = jax.jvp(flux, (latitude, sst), (jnp.asarray([0.0, 1.0, 0.0]), one * 0))
    np.testing.assert_array_equal(forward, 0)
    lat_reverse, sst_reverse = jax.grad(
        lambda lat, temp: flux(lat, temp).sum(), argnums=(0, 1)
    )(latitude, sst)
    assert np.isfinite(lat_reverse).all(), (
        "ERROR undefined dry latitude poisoned pullback"
    )
    assert np.isfinite(sst_reverse).all(), (
        "ERROR undefined dry latitude poisoned temperature pullback"
    )
    np.testing.assert_array_equal(lat_reverse[1], 0)
    np.testing.assert_array_equal(sst_reverse[mask == 0], 0)
