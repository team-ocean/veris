"""Protect native category albedo, source equivalence and surface derivatives."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import veris
from veris import Experiment, Grid, initialize_experiment
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.numerics.surface_temperature import solve4temp
from veris.physics.thermodynamics.growth import Growth


def column(dtype: str = "float64") -> tuple[State, Configuration, PhysicalConstants]:
    """Construct a small real native column with controlled ice and air inputs."""
    state, conf, phys = initialize_experiment(
        Experiment.builtin(
            "growth",
            grid=Grid(2, 3),
            settings={"dtype": dtype, "deltatTherm": 300.0, "deltatDyn": 300.0},
            initial_conditions={
                "Area": 0.5,
                "hIceMean": 0.4,
                "hSnowMean": 0.03,
                "TSurf": 260.0,
            },
        )
    )
    shape = state.Area.shape
    index = jnp.indices(shape)
    state = replace(
        state,
        ATemp=jnp.full_like(state.Area, 263.0),
        aqh=jnp.full_like(state.Area, 0.002),
        LWdown=jnp.full_like(state.Area, 220.0),
        SWdown=jnp.full_like(state.Area, 180.0),
        wSpeed=jnp.full_like(state.Area, 6.0),
        fCori=jnp.where(index[1] % 2 == 0, 1e-4, -1e-4).astype(state.Area.dtype),
    )
    return state, conf, phys


def solver_reference(
    state: State, conf: Configuration, phys: PhysicalConstants, scale: jax.Array
) -> jax.Array:
    """Exercise nonzero radiation and multiple wet/snow branches of the old solver."""
    index = jnp.indices(state.Area.shape)
    ice = jnp.full_like(state.Area, 0.8)
    snow = jnp.take(jnp.asarray([0.0, 0.04, 0.25]), index[1] % 3) * scale
    temperature = jnp.where(index[0] % 2 == 0, 260.0, 274.0)
    freezing = jnp.full_like(state.Area, 271.15)
    return jnp.stack(solve4temp(state, conf, phys, ice, snow, temperature, freezing))


def growth_reference(
    state: State, conf: Configuration, phys: PhysicalConstants, albedo: jax.Array
) -> jax.Array:
    """Retain all native growth outputs, including zero-ice initiation."""
    state = replace(
        state,
        Area=state.Area.at[2, 2].set(0.0),
        hIceMean=state.hIceMean.at[2, 2].set(0.0),
        hSnowMean=state.hSnowMean.at[2, 2].set(0.0),
    )
    return jnp.stack(Growth(state, conf, phys.with_overrides(drySnowAlb=albedo)))


def diagnose(state: State, conf: Configuration, phys: PhysicalConstants) -> jax.Array:
    """Assert a missing public diagnostic as behavior instead of an import error."""
    candidate = getattr(veris, "diagnose_surface_albedo", None)
    assert callable(candidate), "ERROR missing native current-state surface albedo"
    function = cast(
        Callable[[State, Configuration, PhysicalConstants], jax.Array], candidate
    )
    return function(state, conf, phys)


@pytest.mark.parametrize(
    "temperature,coriolis,snow,expected",
    [
        (260.0, 1e-4, 0.0, 0.75),
        (260.0, 1e-4, 0.075, 0.795),
        (260.0, 1e-4, 0.15, 0.84),
        (260.0, 1e-4, 0.3, 0.84),
        (273.15, 1e-4, 0.0, 0.66),
        (274.0, 1e-4, 0.075, 0.68),
        (260.0, -1e-4, 0.0, 0.5),
        (260.0, -1e-4, 0.075, 0.65),
        (273.15, -1e-4, 0.075, 0.4),
    ],
)
def test_surface_albedo_uses_native_snow_temperature_and_hemisphere(
    temperature: float, coriolis: float, snow: float, expected: float
) -> None:
    state, conf, phys = column()
    conf = replace(conf, nITC=1)
    phys = phys.with_overrides(
        dryIceAlb_south=0.5,
        drySnowAlb_south=0.8,
        wetIceAlb_south=0.3,
        wetSnowAlb_south=0.5,
    )
    state = replace(
        state,
        hSnowMean=snow * jnp.sqrt(state.Area**2 + phys.Area_reg),
        TSurf=jnp.full_like(state.Area, temperature),
        fCori=jnp.full_like(state.Area, coriolis),
    )
    result = diagnose(state, conf, phys)
    assert result.shape == state.Area.shape
    np.testing.assert_allclose(result[2:-2, 2:-2], expected, rtol=1e-14)


@pytest.mark.parametrize(
    "categories,expected",
    [
        (1, 0.81),
        (3, (0.77 + 0.81 + 0.84) / 3),
        (5, (0.762 + 0.786 + 0.81 + 0.834 + 0.84) / 5),
    ],
)
def test_category_albedo_averages_native_law(categories: int, expected: float) -> None:
    state, conf, phys = column()
    state = replace(
        state,
        hSnowMean=0.1 * jnp.sqrt(state.Area**2 + phys.Area_reg),
        fCori=jnp.full_like(state.Area, 1e-4),
    )
    result = diagnose(state, replace(conf, nITC=categories), phys)
    np.testing.assert_allclose(result[2:-2, 2:-2], expected, rtol=1e-14)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_surface_albedo_retains_native_dtype(dtype: str) -> None:
    state, conf, phys = column(dtype)
    result = diagnose(state, conf, phys)
    assert result.dtype == jnp.dtype(dtype)


def test_absent_ice_and_dry_nan_do_not_contaminate_albedo_gradients() -> None:
    state, conf, phys = column()
    active = (state.iceMask > 0).at[2, 2].set(False)
    state = replace(
        state,
        Area=jnp.where(active, state.Area, 0.0),
        hIceMean=jnp.where(active, state.hIceMean, 0.0),
        hSnowMean=jnp.where(active, state.hSnowMean, jnp.nan),
        TSurf=jnp.where(active, state.TSurf, jnp.nan),
        fCori=jnp.where(active, state.fCori, jnp.nan),
    )
    phys = phys.with_overrides(Area_reg=0.0)

    def loss(albedo: jax.Array) -> jax.Array:
        return diagnose(state, conf, phys.with_overrides(drySnowAlb=albedo)).sum()

    output = diagnose(state, conf, phys)
    np.testing.assert_array_equal(jnp.where(active, 0, output), 0)
    value = jnp.asarray(0.84)
    _, tangent = jax.jvp(jax.jit(loss), (value,), (jnp.ones_like(value),))
    gradient = jax.grad(loss)(value)
    assert bool(jnp.isfinite(tangent) & jnp.isfinite(gradient))
    assert float(gradient) > 0
    np.testing.assert_allclose(tangent, gradient, rtol=1e-13)
    finite_difference = (loss(value + 1e-4) - loss(value - 1e-4)) / 2e-4
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-9)


def test_surface_extraction_preserves_solver_outputs_and_jvp() -> None:
    reference = (
        Path(__file__).parents[1] / "reference" / "surface_properties_baseline.npz"
    )
    with np.load(reference, allow_pickle=False) as baseline:
        assert (
            str(baseline["native_base"]) == "6c205a356dd1b25c26430f899cb8998523ab3447"
        )
        for name, function, value in (
            ("solver", solver_reference, 1.0),
            ("growth", growth_reference, 0.84),
        ):
            point = jnp.asarray(value)
            evaluate = partial(function, *column())
            output, tangent = jax.jvp(evaluate, (point,), (jnp.ones_like(point),))
            np.testing.assert_allclose(output, baseline[name], rtol=0, atol=0)
            np.testing.assert_allclose(tangent, baseline[name + "_jvp"], rtol=0, atol=0)
