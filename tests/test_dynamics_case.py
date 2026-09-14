"""Reference snapshot geometry and independent dynamics-driver composition."""

from dataclasses import replace
from importlib import import_module
from types import ModuleType

import jax
import numpy as np
import pytest


def case() -> ModuleType:
    """Import lazily so missing implementation reports an explicit test failure."""
    try:
        return import_module("veris.setups.run_dyn")
    except ModuleNotFoundError:
        pytest.fail("dynamics case is not implemented")


def test_reference_initial_fields_and_wind() -> None:
    """A shifted/rotated wind or swapped face mask must change this oracle."""
    state, conf, phys = case().initialize(8, 10)
    assert conf.nx == 8 and conf.ny == 10 and not conf.use_sharding
    assert conf.nEVPsteps == 120 and conf.useAdaptiveEVP
    assert conf.deltatDyn == 600 and not conf.useRelativeWind
    assert phys.rhoIce > 0
    x, y = np.meshgrid(
        (np.arange(8) + 0.5) * 512000 / 7,
        (np.arange(10) + 0.5) * 512000 / 9,
        indexing="ij",
    )
    xx, yy = x - 352000, y - 352000
    factor = -15 * np.exp(-np.hypot(xx, yy) / 100000) / 50000
    np.testing.assert_allclose(
        state.uWind[2:-2, 2:-2],
        factor * (np.cos(0.4 * np.pi) * xx + np.sin(0.4 * np.pi) * yy),
    )
    np.testing.assert_allclose(
        state.vWind[2:-2, 2:-2],
        factor * (-np.sin(0.4 * np.pi) * xx + np.cos(0.4 * np.pi) * yy),
    )
    mask = np.ones((8, 10))
    mask[-1, :] = 0
    mask[:, -1] = 0
    for name, expected in [
        ("maskInC", mask),
        ("maskInU", mask * np.roll(mask, 1, 0)),
        ("maskInV", mask * np.roll(mask, 1, 1)),
    ]:
        np.testing.assert_array_equal(
            getattr(state, name), np.pad(expected, 2, mode="wrap")
        )
    np.testing.assert_array_equal(state.hIceMean, 0.3)
    np.testing.assert_array_equal(state.Area, 1.0)
    np.testing.assert_array_equal(state.hSnowMean, 0.0)
    np.testing.assert_allclose(
        state.uOcean[2:-2, 2:-2],
        0.01 * (2 * y - 512000) / 512000 * mask * np.roll(mask, 1, 0),
    )
    np.testing.assert_allclose(
        state.fCori[2:-2, 2:-2],
        np.broadcast_to(np.linspace(1.4604e-4, 1.4596e-4 + 8e-8 * 10, 10), (8, 10)),
    )


def test_dynamics_matches_reference_kernel_sequence() -> None:
    """Every evolving field and returned stress must match the notebook order."""
    from veris.advection import Advection
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.clean_up import clean_up_advection, ridging
    from veris.dynamics_routines import SeaIceStrength
    from veris.dynsolver import IceVelocities, WindForcingXY
    from veris.fill_overlap import fill_overlap
    from veris.ocean_stress import OceanStressUV

    module = case()
    initial, conf, phys = module.initialize(6, 8, settings_overrides={"nEVPsteps": 2})
    expected = initial
    for names, kernel in [
        ("SeaIceMassC SeaIceMassU SeaIceMassV", SeaIceMass),
        ("AreaW AreaS", AreaWS),
        ("WindForcingX WindForcingY", WindForcingXY),
    ]:
        expected = replace(
            expected,
            **dict(zip(names.split(), kernel(expected, conf, phys), strict=True)),
        )
    expected = replace(expected, SeaIceStrength=SeaIceStrength(expected, conf, phys))
    expected = replace(
        expected,
        **dict(
            zip(
                ["uIce", "vIce", "sigma1", "sigma2", "sigma12"],
                IceVelocities(expected, conf, phys),
                strict=True,
            )
        ),
    )
    stress = OceanStressUV(expected, conf, phys)
    expected = replace(
        expected,
        **dict(
            zip(
                ["hIceMean", "hSnowMean", "Area"],
                Advection(expected, conf, phys),
                strict=True,
            )
        ),
    )
    expected = replace(
        expected,
        **dict(
            zip(
                [
                    "hIceMean",
                    "hSnowMean",
                    "Area",
                    "TSurf",
                    "os_hIceMean",
                    "os_hSnowMean",
                ],
                clean_up_advection(expected, conf, phys),
                strict=True,
            )
        ),
    )
    expected = replace(expected, Area=ridging(expected, conf, phys))
    expected = jax.tree.map(lambda a: fill_overlap(a, conf), expected)
    actual, diagnostics = module.step_with_diagnostics(initial, conf, phys)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(diagnostics.OceanStressU, fill_overlap(stress[0], conf))
    np.testing.assert_array_equal(actual.Qnet, initial.Qnet)
    compiled = module.compiled_step(initial, conf, phys)
    for a, b in zip(jax.tree.leaves(compiled), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("nx,ny", [(1, 8), (8, 1), (True, 8), (8, 2.5)])
def test_invalid_grid_rejected(nx: int, ny: float) -> None:
    """Grid validation occurs before field allocation."""
    with pytest.raises((ValueError, TypeError)):
        case().initialize(nx, ny)


def test_dynamics_wind_sensitivity_matches_finite_difference() -> None:
    """The public composition retains AD through forcing and EVP dynamics."""
    import jax.numpy as jnp

    module = case()
    from test_evp_optimization import oracle_state

    # This smooth oracle complements the stationary, coastal zero-strain
    # and full-State pullback regressions in test_ad_zero_states.py.
    state, conf, phys = oracle_state(False)
    conf = replace(conf, nEVPsteps=2)

    def objective(scale: jax.Array) -> jax.Array:
        forced = replace(state, uWind=state.uWind * scale, vWind=state.vWind * scale)
        result = module.compiled_step(forced, conf, phys)
        return jnp.sum(result.uIce[2:-2, 2:-2] ** 2 + result.vIce[2:-2, 2:-2] ** 2)

    value = jnp.asarray(1.0)
    derivative = jax.grad(objective)(value)
    delta = 1e-3
    finite_difference = (objective(value + delta) - objective(value - delta)) / (
        2 * delta
    )
    assert abs(float(derivative)) > 1e-10
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-5, atol=1e-10)
