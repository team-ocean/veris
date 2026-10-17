"""Shared timesteps preserve kernel physics and dynamic forcing derivatives.

These tests cover orchestration rather than re-testing kernel equations. The
independent kernel compositions are all-field numerical oracles; padded
physical forcing catches missing halo preparation before staggered stencils.
"""

from dataclasses import fields, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris import Experiment, Grid, initialize_experiment
from veris.execution.advance import advance
from veris.grid.halos import fill_state_overlap
from veris.model.configuration import Configuration
from veris.model.diagnostics import Diagnostics
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.physics.dynamics.driver import dynamics_transport
from veris.physics.thermodynamics.growth import Growth


def kernel_reference(
    state: State,
    conf: Configuration,
    phys: PhysicalConstants,
    physics: str,
) -> tuple[State, Diagnostics]:
    """Compose numerical stages without the public forcing or timestep wrapper."""
    zeros = jnp.zeros_like(state.Area)
    stress_u = stress_v = zeros
    if physics != "thermodynamics":
        state, stress_u, stress_v = dynamics_transport(state, conf, phys)
    freshwater = salt = penetrating = zeros
    if physics != "dynamics":
        values = Growth(state, conf, phys)
        state = replace(
            state,
            hIceMean=values[0],
            hSnowMean=values[1],
            Area=values[2],
            TSurf=values[3],
            Qsw=values[6],
            Qnet=values[7],
            SeaIceLoad=values[8],
            recip_hIceMean=values[10],
        )
        freshwater, salt, penetrating = values[4], values[5], values[9]
    diagnostics = Diagnostics(penetrating, stress_u, stress_v, freshwater, salt, zeros)
    if physics != "thermodynamics" or not conf.enable_cyclic_y:
        state = fill_state_overlap(state, conf)
        diagnostics = fill_state_overlap(diagnostics, conf)
    return state, diagnostics


def assert_pair_close(
    actual: tuple[State, Diagnostics], expected: tuple[State, Diagnostics]
) -> None:
    """Compare every named calculation and coupling array without array dumps."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for result, reference in zip(actual, expected, strict=True):
        for item in fields(reference):
            left = np.asarray(getattr(result, item.name))
            right = np.asarray(getattr(reference, item.name))
            assert left.shape == right.shape, f"ERROR {item.name} shape mismatch"
            assert left.dtype == right.dtype, f"ERROR {item.name} dtype mismatch"
            assert np.isfinite(left).all() and np.isfinite(right).all(), (
                f"ERROR {item.name} nonfinite output"
            )
            error = np.abs(left - right)
            tolerance = 3e-5 if right.dtype == np.dtype("float32") else 2e-11
            assert np.all(error <= tolerance * (1 + np.abs(right))), (
                f"ERROR {item.name} max absolute difference {error.max():.6g}"
            )


def pad_atmosphere(value: np.ndarray, cyclic_y: bool) -> jax.Array:
    """Build atmospheric halos independently of the production fill helpers."""
    wrapped_x = np.pad(value, ((2, 2), (0, 0)), mode="wrap")
    padded = np.pad(wrapped_x, ((0, 0), (2, 2)), mode="wrap" if cyclic_y else "edge")
    return jnp.asarray(padded, dtype=jnp.float64)


@pytest.mark.parametrize("physics", ["dynamics", "thermodynamics", "coupled"])
@pytest.mark.parametrize("cyclic_y", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_public_timestep_matches_every_kernel_state_and_diagnostic(
    physics: str, cyclic_y: bool, dtype: str
) -> None:
    """Catch reordered stress/transport/growth, altered flux policy or final halos."""
    options = {
        "dtype": dtype,
        "nEVPsteps": 2,
        "deltatDyn": 600.0,
        "deltatTherm": 600.0,
        "enable_cyclic_y": cyclic_y,
    }
    forcing = None
    if physics == "dynamics":
        initial, conf, phys = initialize_experiment(
            Experiment.builtin("dynamics", grid=Grid(4, 5), settings=options)
        )
        expected = kernel_reference(initial, conf, phys, physics)
    elif physics == "thermodynamics":
        initial, conf, phys = initialize_experiment(
            Experiment.builtin("growth", settings=options)
        )
        expected = kernel_reference(initial, conf, phys, physics)
    else:
        initial, conf, phys = initialize_experiment(
            Experiment.builtin("island", grid=Grid(4, 5), settings=options)
        )
        initial = replace(
            initial,
            Qnet=jnp.full_like(initial.Qnet, -777.0),
            Qsw=jnp.full_like(initial.Qsw, 888.0),
        )
        forcing = {
            "Qnet": jnp.array(85.0, dtype=dtype),
            "Qsw": jnp.array(0.0, dtype=dtype),
        }
        expected = kernel_reference(
            replace(
                initial,
                Qnet=jnp.full_like(initial.Qnet, 85.0),
                Qsw=jnp.full_like(initial.Qsw, 0.0),
            ),
            conf,
            phys,
            physics,
        )
    actual = advance(initial, conf, phys, forcing, physics=physics)
    assert_pair_close(actual, expected)


@pytest.mark.parametrize("cyclic_y", [False, True])
@pytest.mark.parametrize("component", ["uWind", "vWind"])
def test_spatial_wind_prepares_halos_and_derives_speed(
    cyclic_y: bool, component: str
) -> None:
    """Catch staggered forcing using stale halos or thermodynamics using old speed."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin(
            "dynamics",
            grid=Grid(4, 5),
            settings={"nEVPsteps": 2, "enable_cyclic_y": cyclic_y},
        )
    )
    x, y = np.indices((4, 5))
    prescribed = 3.0 + 0.7 * x + 0.3 * y
    padded = pad_atmosphere(prescribed, cyclic_y)
    forced = replace(initial, **{component: padded})
    forced = replace(forced, wSpeed=jnp.sqrt(forced.uWind**2 + forced.vWind**2))
    expected = kernel_reference(forced, conf, phys, "dynamics")
    actual = advance(
        initial,
        conf,
        phys,
        {component: jnp.asarray(prescribed)},
        physics="dynamics",
    )
    assert_pair_close(actual, expected)
    reference_speed = np.hypot(
        np.asarray(forced.uWind)[2:-2, 2:-2],
        np.asarray(forced.vWind)[2:-2, 2:-2],
    )
    np.testing.assert_allclose(actual[0].wSpeed[2:-2, 2:-2], reference_speed)


@pytest.mark.parametrize("component", ["uOcean", "vOcean"])
def test_closed_y_ocean_forcing_prepares_velocity_halos_before_dynamics(
    component: str,
) -> None:
    """Catch using atmospheric edge halos or retaining the owned south normal flow."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin(
            "dynamics",
            grid=Grid(4, 5),
            settings={"nEVPsteps": 2, "enable_cyclic_y": False},
        )
    )
    x, y = np.indices((4, 5))
    prescribed = 0.02 + 0.003 * x + 0.002 * y
    # Independently construct x-periodic, dry-exterior ocean-velocity storage.
    stored = np.zeros((8, 9))
    x_indices = (np.arange(8) - 2) % 4
    stored[:, 2:-2] = prescribed[x_indices, :]
    if component == "vOcean":
        stored[:, 2] = 0.0
    expected = kernel_reference(
        replace(initial, **{component: jnp.asarray(stored)}), conf, phys, "dynamics"
    )
    actual = advance(
        initial,
        conf,
        phys,
        {component: jnp.asarray(prescribed)},
        physics="dynamics",
    )
    assert_pair_close(actual, expected)
    actual_ocean = getattr(actual[0], component)
    np.testing.assert_array_equal(actual_ocean, stored)
    np.testing.assert_array_equal(actual_ocean[:, :2], 0.0)
    np.testing.assert_array_equal(actual_ocean[:, -2:], 0.0)
    if component == "vOcean":
        np.testing.assert_array_equal(actual_ocean[:, 2], 0.0)
        assert np.any(prescribed[:, 0] != 0), "ERROR fixture must expose south closure"


def test_explicit_speed_takes_precedence_over_component_derivation() -> None:
    """Catch replacing a separately prescribed thermodynamic wind magnitude."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin("island", grid=Grid(4, 5), settings={"nEVPsteps": 2})
    )
    expected = replace(
        initial,
        uWind=jnp.full_like(initial.uWind, 3.0),
        vWind=jnp.full_like(initial.vWind, 4.0),
        wSpeed=jnp.full_like(initial.wSpeed, 7.0),
        Qnet=jnp.full_like(initial.Qnet, 100.0),
        Qsw=jnp.full_like(initial.Qsw, 0.0),
    )
    actual = advance(
        initial,
        conf,
        phys,
        {"uWind": 3.0, "vWind": 4.0, "wSpeed": 7.0, "Qnet": 100.0, "Qsw": 0.0},
    )
    assert_pair_close(actual, kernel_reference(expected, conf, phys, "coupled"))
    np.testing.assert_array_equal(actual[0].wSpeed, 7.0)


def test_unprescribed_wind_speed_keeps_independent_state_value() -> None:
    """Catch deriving wind magnitude when no wind components were supplied."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin("growth", settings={"deltatTherm": 600.0})
    )
    actual = advance(initial, conf, phys, {"Qnet": 95.0}, physics="thermodynamics")
    expected = kernel_reference(
        replace(initial, Qnet=jnp.full_like(initial.Qnet, 95.0)),
        conf,
        phys,
        "thermodynamics",
    )
    assert_pair_close(actual, expected)
    np.testing.assert_array_equal(actual[0].wSpeed, initial.wSpeed)


def test_prescribed_heat_is_restored_after_growth_overwrites_it() -> None:
    """Catch applying constant atmospheric heat only on the first timestep."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin("growth", settings={"deltatTherm": 600.0})
    )
    forcing = {"Qnet": jnp.array(110.0), "Qsw": jnp.array(6.0)}
    actual = expected = initial
    for _ in range(3):
        expected_pair = kernel_reference(
            replace(
                expected,
                Qnet=jnp.full_like(expected.Qnet, 110.0),
                Qsw=jnp.full_like(expected.Qsw, 6.0),
            ),
            conf,
            phys,
            "thermodynamics",
        )
        actual_pair = advance(actual, conf, phys, forcing, physics="thermodynamics")
        assert_pair_close(actual_pair, expected_pair)
        actual, expected = actual_pair[0], expected_pair[0]
    assert not np.allclose(actual.Qnet, 110.0), (
        "ERROR flux fixture must expose overwrite"
    )


def test_absent_heat_forcing_preserves_recursive_growth_fluxes() -> None:
    """Catch unconditionally restoring default heat fluxes for the growth preset."""
    initial, conf, phys = initialize_experiment(Experiment.builtin("growth"))
    actual = expected = reset = initial
    for _ in range(3):
        expected_pair = kernel_reference(expected, conf, phys, "thermodynamics")
        actual_pair = advance(actual, conf, phys, physics="thermodynamics")
        reset = kernel_reference(
            replace(reset, Qnet=initial.Qnet, Qsw=initial.Qsw),
            conf,
            phys,
            "thermodynamics",
        )[0]
        assert_pair_close(actual_pair, expected_pair)
        actual, expected = actual_pair[0], expected_pair[0]
    assert not np.allclose(actual.Qnet, reset.Qnet), (
        "ERROR recursive fixture must distinguish reset forcing"
    )


@pytest.mark.parametrize("cyclic_y", [False, True])
def test_spatial_heat_fields_use_global_physical_grid(cyclic_y: bool) -> None:
    """Catch scalar-only heat handling, axis swaps or ignoring prescribed shortwave."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin(
            "growth",
            grid=Grid(4, 5),
            settings={"deltatTherm": 600.0, "enable_cyclic_y": cyclic_y},
        )
    )
    x, y = np.indices((4, 5))
    heat = 70.0 + 5.0 * x + 3.0 * y
    shortwave = 2.0 + x + 0.5 * y
    expected = kernel_reference(
        replace(
            initial,
            Qnet=pad_atmosphere(heat, cyclic_y),
            Qsw=pad_atmosphere(shortwave, cyclic_y),
        ),
        conf,
        phys,
        "thermodynamics",
    )
    actual = advance(
        initial,
        conf,
        phys,
        {"Qnet": jnp.asarray(heat), "Qsw": jnp.asarray(shortwave)},
        physics="thermodynamics",
    )
    assert_pair_close(actual, expected)


def smooth_coupled_case() -> tuple[State, Configuration, PhysicalConstants]:
    """Keep concentration below ridging and temperature below melting thresholds."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin(
            "island",
            grid=Grid(4, 5),
            settings={"nEVPsteps": 2, "useAdaptiveEVP": False},
        )
    )
    x, y = np.indices((4, 5))
    a = pad_atmosphere(
        np.sin(2 * np.pi * x / 4) + 0.3 * np.cos(2 * np.pi * y / 5), True
    )
    b = pad_atmosphere(np.cos(2 * np.pi * (x / 4 + y / 5)), True)
    ones = jnp.ones_like(initial.Area)
    ice = 1.1 + 0.12 * a
    snow = 0.06 * ones
    initial = replace(
        initial,
        iceMask=ones,
        maskInC=ones,
        iceMaskU=ones,
        maskInU=ones,
        iceMaskV=ones,
        maskInV=ones,
        hIceMean=ice,
        hSnowMean=snow,
        Area=0.78 + 0.025 * b,
        uIce=0.05 + 0.012 * a,
        vIce=0.03 + 0.013 * b,
        uOcean=-0.01 + 0.003 * b,
        vOcean=-0.02 + 0.004 * a,
        sigma1=-700.0 + 35.0 * a,
        sigma2=80.0 + 12.0 * b,
        sigma12=25.0 + 3.0 * a,
        SeaIceLoad=phys.rhoIce * ice + phys.rhoSnow * snow,
        recip_hIceMean=1 / jnp.sqrt(ice**2 + phys.hIce_reg),
    )
    return initial, conf, phys


def test_jitted_coupled_state_and_spatial_forcing_grad_jvp_vjp_fd() -> None:
    """Catch host conversions, static forcing, stopped gradients or changed tangents."""
    initial, conf, phys = smooth_coupled_case()
    x, y = jnp.indices((4, 5), dtype=jnp.float64)
    heat = 90.0 + 2.0 * x + 3.0 * y
    heat_direction = 70.0 + 10.0 * jnp.sin(x + 2.0 * y)
    ice_direction = pad_atmosphere(np.asarray(0.3 + 0.07 * jnp.cos(2.0 * x - y)), True)
    weights = 1.0 + 0.2 * jnp.sin(x + 3.0 * y)

    def objective(ice: jax.Array, cooling: jax.Array) -> jax.Array:
        result, diagnostic = advance(
            replace(initial, hIceMean=ice),
            conf,
            phys,
            {"Qnet": cooling, "Qsw": jnp.array(0.0)},
        )
        quantity = (
            result.hIceMean
            + 10.0 * result.uIce
            + diagnostic.EmPmR
            + diagnostic.OceanStressU
        )
        return jnp.mean(weights * quantity[2:-2, 2:-2])

    function = jax.jit(objective)
    point = (initial.hIceMean, heat)
    np.testing.assert_allclose(function(*point), objective(*point), rtol=2e-11)
    value, pullback = jax.vjp(function, *point)
    reverse = pullback(jnp.ones_like(value))
    gradients = jax.grad(function, argnums=(0, 1))(*point)
    for actual, expected in zip(gradients, reverse, strict=True):
        assert np.isfinite(actual).all(), "ERROR nonfinite coupled gradient"
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    directions = (
        (ice_direction, jnp.zeros_like(heat)),
        (jnp.zeros_like(initial.hIceMean), heat_direction),
    )
    epsilon = 1e-4
    for direction in directions:
        _, tangent = jax.jvp(function, point, direction)
        projected = sum(
            jnp.vdot(gradient, delta)
            for gradient, delta in zip(reverse, direction, strict=True)
        )
        plus = tuple(p + epsilon * d for p, d in zip(point, direction, strict=True))
        minus = tuple(p - epsilon * d for p, d in zip(point, direction, strict=True))
        finite_difference = (function(*plus) - function(*minus)) / (2 * epsilon)
        assert abs(float(tangent)) > 1e-9, "ERROR missing initial/forcing sensitivity"
        np.testing.assert_allclose(tangent, projected, rtol=3e-5, atol=1e-7)
        np.testing.assert_allclose(tangent, finite_difference, rtol=2e-4, atol=2e-8)


def test_dynamic_wind_component_derivatives_include_derived_thermodynamic_speed() -> (
    None
):
    """Catch static wind conversion, detached magnitude or changed momentum tangents."""
    initial, conf, phys = smooth_coupled_case()
    x, y = jnp.indices((4, 5), dtype=jnp.float64)
    wind = 2.0 + 0.1 * x + 0.05 * y
    direction = 0.7 + 0.1 * jnp.cos(x + 2.0 * y)
    weights = 1.0 + 0.2 * jnp.sin(x + 3.0 * y)

    def loss(result: State, diagnostic: Diagnostics) -> jax.Array:
        quantity = (
            result.hIceMean
            + 10.0 * result.uIce
            + diagnostic.OceanStressU
            + 0.1 * result.wSpeed
        )
        return jnp.mean(weights * quantity[2:-2, 2:-2])

    def objective(value: jax.Array) -> jax.Array:
        result, diagnostic = advance(
            initial,
            conf,
            phys,
            {"vWind": value, "Qnet": jnp.array(85.0), "Qsw": jnp.array(0.0)},
        )
        return loss(result, diagnostic)

    def kernel_objective(value: jax.Array) -> jax.Array:
        # JAX padding preserves the independent oracle's forcing derivative.
        stored = jnp.pad(value, ((2, 2), (2, 2)), mode="wrap")
        forced = replace(
            initial,
            vWind=stored,
            wSpeed=jnp.sqrt(initial.uWind**2 + stored**2),
            Qnet=jnp.full_like(initial.Qnet, 85.0),
            Qsw=jnp.full_like(initial.Qsw, 0.0),
        )
        result, diagnostic = kernel_reference(forced, conf, phys, "coupled")
        return loss(result, diagnostic)

    function = jax.jit(objective)
    reference = jax.jit(kernel_objective)
    np.testing.assert_allclose(function(wind), reference(wind), rtol=2e-11, atol=2e-11)
    gradient = jax.grad(function)(wind)
    np.testing.assert_allclose(
        gradient, jax.grad(reference)(wind), rtol=3e-5, atol=1e-7
    )
    assert np.isfinite(gradient).all(), "ERROR nonfinite dynamic wind gradient"
    _, tangent = jax.jvp(function, (wind,), (direction,))
    projected = jnp.vdot(gradient, direction)
    epsilon = 1e-4
    finite_difference = (
        function(wind + epsilon * direction) - function(wind - epsilon * direction)
    ) / (2 * epsilon)
    assert abs(float(tangent)) > 1e-8, "ERROR missing wind sensitivity"
    np.testing.assert_allclose(tangent, projected, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-4, atol=2e-8)


@pytest.mark.parametrize(
    ("forcing", "physics", "message"),
    [
        ({"hIceMean": 1.0}, "coupled", "field"),
        ({"Qnet": jnp.ones((4, 6))}, "coupled", "shape"),
        (None, "unknown", "physics"),
    ],
)
def test_invalid_pure_timestep_definition_is_informative(
    forcing: Any, physics: str, message: str
) -> None:
    """Catch unsupported evolved-field forcing, wrong grid shape or ignored mode."""
    initial, conf, phys = initialize_experiment(
        Experiment.builtin("island", grid=Grid(4, 5), settings={"nEVPsteps": 2})
    )
    with pytest.raises((TypeError, ValueError), match=message):
        advance(initial, conf, phys, forcing, physics=physics)


@pytest.mark.parametrize("component", ["uWind", "vWind"])
def test_calm_prescribed_wind_uses_zero_norm_linearization(component: str) -> None:
    """Match the maintained symmetric-difference convention at a calm wind."""
    initial, conf, phys = initialize_experiment(Experiment.builtin("growth"))

    def objective(wind: jax.Array) -> jax.Array:
        forcing = {"uWind": jnp.array(0.0), "vWind": jnp.array(0.0)}
        forcing[component] = wind
        state, _ = advance(initial, conf, phys, forcing, physics="thermodynamics")
        return state.wSpeed.mean()

    function = jax.jit(objective)
    point = jnp.array(0.0)
    assert function(point) == 0
    gradient = jax.grad(function)(point)
    _, tangent = jax.jvp(function, (point,), (jnp.array(1.0),))
    value, pullback = jax.vjp(function, point)
    reverse = pullback(jnp.ones_like(value))[0]
    epsilon = 1e-4
    finite_difference = (function(point + epsilon) - function(point - epsilon)) / (
        2 * epsilon
    )
    np.testing.assert_array_equal(
        [gradient, tangent, reverse, finite_difference], np.zeros(4)
    )


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "x64"])
def test_state_metadata_mismatch_fails_before_physics(mismatch: str) -> None:
    """Reject inconsistent array storage rather than silently changing precision."""
    state, conf, phys = initialize_experiment(Experiment.builtin("growth"))
    if mismatch == "shape":
        conf = replace(conf, nx=4, ny=5)
    elif mismatch == "dtype":
        conf, phys = replace(conf, dtype="float32"), replace(phys, dtype="float32")
    if mismatch == "x64":
        with (
            jax.enable_x64(False),
            pytest.raises(ValueError, match="x64|dtype|precision"),
        ):
            advance(state, conf, phys, physics="thermodynamics")
    else:
        with pytest.raises(ValueError, match="shape|dtype|precision"):
            advance(state, conf, phys, physics="thermodynamics")


def test_stable_wind_norm_keeps_large_finite_components_and_derivatives() -> None:
    """Prevent the calm-wind fix from introducing overflow through squaring."""
    from veris.numerics import autodiff as _ad

    wind = jnp.array([3e200, 4e200])

    def magnitude(value: jax.Array) -> jax.Array:
        return _ad.norm_hypot(value[0], value[1])

    np.testing.assert_allclose(magnitude(wind), 5e200, rtol=1e-14)
    np.testing.assert_allclose(jax.grad(magnitude)(wind), [0.6, 0.8], rtol=1e-14)
