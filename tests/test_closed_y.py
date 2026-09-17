"""Closed meridional walls must block transport without losing interior cells."""

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.advection import calc_Advection
from veris.configuration import Configuration
from veris.fill_overlap import fill_overlap, fill_overlap_uv
from veris.initialization import initialize


@pytest.mark.parametrize("ny", [2, 5])
def test_closed_halos_and_normal_wall_velocity(ny: int) -> None:
    """Catch accidental wrap, wrong face index and loss of edge tracer cells."""
    conf = Configuration(use_sharding=False, enable_cyclic_y=False)
    interior = np.arange(3 * ny, dtype=float).reshape(3, ny) + 1
    data = jnp.asarray(np.pad(interior, 2, constant_values=-999))
    expected = np.pad(
        np.pad(interior, ((2, 2), (0, 0)), mode="wrap"), ((0, 0), (2, 2)), mode="edge"
    )
    np.testing.assert_array_equal(fill_overlap(data, conf), expected)
    u, v = fill_overlap_uv(data, data, conf)
    expected_u = expected.copy()
    expected_u[:, :2] = expected_u[:, -2:] = 0
    expected_v = expected_u.copy()
    expected_v[:, 2] = 0
    np.testing.assert_array_equal(u, expected_u)
    np.testing.assert_array_equal(v, expected_v)


@pytest.mark.parametrize("speed", [-0.2, 0.2])
def test_closed_transport_conserves_mass_and_its_derivative(speed: float) -> None:
    """Outward forcing cannot remove mass or couple opposite global y edges."""
    shape = (9, 11)
    ones = jnp.ones(shape)
    state, conf, phys = initialize(
        5,
        7,
        settings_overrides={
            "use_sharding": False,
            "enable_cyclic_y": False,
            "deltatTherm": 1.0,
        },
        state_overrides={"vIce": speed * ones, "dxG": ones, "dyG": ones},
    )
    for name in ("iceMask", "iceMaskU", "maskInC", "maskInU"):
        mask = np.asarray(getattr(state, name))
        np.testing.assert_array_equal(mask[:, :2], 0)
        np.testing.assert_array_equal(mask[:, -2:], 0)
        np.testing.assert_array_equal(mask[2:-2, 2:-2], 1)
    for name in ("iceMaskV", "maskInV", "vIce"):
        value = np.asarray(getattr(state, name))
        np.testing.assert_array_equal(value[:, :3], 0)
        np.testing.assert_array_equal(value[:, -2:], 0)

    # Unequal successive slopes avoid the Superbee kink at slope ratio one.
    field = jnp.asarray(1 + (np.arange(35).reshape(5, 7) / 35) ** 2)

    def evolve(values: jax.Array) -> jax.Array:
        padded = fill_overlap(jnp.pad(values, 2), conf)
        return calc_Advection(replace(state, hIceMean=padded), conf, phys, padded)[
            2:-2, 2:-2
        ]

    np.testing.assert_allclose(evolve(field).sum(), field.sum(), atol=1e-12)
    direction = jnp.cos(field)
    _, tangent = jax.jvp(evolve, (field,), (direction,))
    np.testing.assert_allclose(tangent.sum(), direction.sum(), atol=1e-12)
    np.testing.assert_allclose(
        jax.grad(lambda x: evolve(x).sum())(field), 1, atol=1e-12
    )
    weights = jnp.sin(jnp.arange(35).reshape(5, 7))
    objective = lambda x: jnp.sum(evolve(x) * weights)
    fd = (
        objective(field + 1e-5 * direction) - objective(field - 1e-5 * direction)
    ) / 2e-5
    np.testing.assert_allclose(jnp.sum(tangent * weights), fd, atol=1e-8)


def test_cyclic_y_requires_boolean() -> None:
    """Reject ambiguous static boundary switches before compilation."""
    with pytest.raises(TypeError, match="enable_cyclic_y"):
        Configuration(enable_cyclic_y=cast(bool, "false"))


def test_four_cpu_closed_coupled_values_and_ad() -> None:
    """Exercise real communication and wall dynamics in a fresh four-device process."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "tests/closed_y_sharding_probe.py")],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            JAX_NUM_CPU_DEVICES="4",
            PYTHONPATH=str(root),
        ),
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR closed-y distributed probe: {result.stdout[-1000:]} {result.stderr[-2000:]}"
    )
    assert result.stdout.count("two steps, all fields and AD pass") == 2


def test_closed_allocation_without_mesh_retains_execution_setting() -> None:
    """Host allocation must not require an execution mesh to close boundaries."""
    state, conf, _ = initialize(settings_overrides={"enable_cyclic_y": False})
    assert conf.use_sharding
    np.testing.assert_array_equal(state.iceMaskV[:, 2], 0)


def test_growth_setup_keeps_closed_exterior_dry() -> None:
    """The prescribed column must apply walls after its initial-field overrides."""
    from veris.setups import run_growth

    state, conf, phys = run_growth.initialize(
        settings_overrides={"enable_cyclic_y": False}
    )
    for result in (state, run_growth.step(state, conf, phys)):
        for name in ("hIceMean", "hSnowMean", "Area"):
            np.testing.assert_array_equal(getattr(result, name)[:, :2], 0)
            np.testing.assert_array_equal(getattr(result, name)[:, -2:], 0)


@pytest.mark.parametrize("speed", [-0.2, 0.2])
def test_closed_intensive_transport_preserves_constant(speed: float) -> None:
    """The transport-divergence correction must use the same closed wall flux."""
    state, conf, phys = initialize(
        5,
        7,
        settings_overrides={
            "use_sharding": False,
            "enable_cyclic_y": False,
            "extensiveFld": False,
            "deltatTherm": 1.0,
        },
        state_overrides={
            "vIce": jnp.full((9, 11), speed),
            "dxG": jnp.ones((9, 11)),
            "dyG": jnp.ones((9, 11)),
            "recip_hIceMean": jnp.ones((9, 11)),
        },
    )
    field = jnp.full((9, 11), 1.7)
    result = calc_Advection(state, conf, phys, field)
    np.testing.assert_allclose(result[2:-2, 2:-2], 1.7, atol=1e-14)


@pytest.mark.parametrize("no_slip", [True, False])
def test_closed_coupled_steps_preserve_walls_and_finite_ad(no_slip: bool) -> None:
    """Setup overrides and repeated refreshes cannot revive dry exterior masks."""
    from veris.setups import island

    state, conf, phys = island.initialize(
        nx=6,
        ny=8,
        settings_overrides={
            "enable_cyclic_y": False,
            "noSlip": no_slip,
            "nEVPsteps": 2,
        },
    )
    for vs in (state, island.step(island.step(state, conf, phys), conf, phys)):
        for name in (
            "iceMask",
            "iceMaskU",
            "maskInC",
            "maskInU",
            "hIceMean",
            "Area",
            "uIce",
        ):
            value = np.asarray(getattr(vs, name))
            np.testing.assert_array_equal(value[:, :2], 0)
            np.testing.assert_array_equal(value[:, -2:], 0)
        for name in ("iceMaskV", "maskInV", "vIce"):
            value = np.asarray(getattr(vs, name))
            np.testing.assert_array_equal(value[:, :3], 0)
            np.testing.assert_array_equal(value[:, -2:], 0)
        for value in jax.tree.leaves(vs):
            assert np.isfinite(value).all(), "ERROR nonfinite closed-wall state"
        assert np.all(np.asarray(vs.iceMask)[2:-2, 2] == 1)
        assert np.all(np.asarray(vs.iceMask)[2:-2, -3] == 1)

    def loss(wind: jax.Array) -> jax.Array:
        initial = replace(state, vWind=state.vWind + wind)
        result = island.step(island.step(initial, conf, phys), conf, phys)
        return jnp.sum(result.hIceMean[2:-2, 2:-2] * jnp.arange(48).reshape(6, 8))

    x = jnp.asarray(0.7)
    _, tangent = jax.jvp(loss, (x,), (jnp.ones_like(x),))
    reverse = jax.grad(loss)(x)
    fd = (loss(x + 1e-3) - loss(x - 1e-3)) / 2e-3
    assert np.isfinite(tangent) and np.isfinite(reverse)
    np.testing.assert_allclose(tangent, reverse, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(tangent, fd, rtol=2e-3, atol=1e-7)
