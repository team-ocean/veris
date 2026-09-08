"""Transport tests use exact donor translations and periodic volume budgets."""

import importlib

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def transport(halo):
    """Import transport after selecting real serial halo exchange."""
    return importlib.import_module("veris.advection")


@pytest.fixture
def transport_state(state):
    """Unit metric periodic grid with a rectangular 5 by 7 interior."""

    def build(u, v):
        ones = np.ones((9, 11))
        return state(
            **{
                name: ones
                for name in (
                    "dyG",
                    "dxG",
                    "iceMaskU",
                    "iceMaskV",
                    "iceMask",
                    "maskInC",
                    "maskInU",
                    "maskInV",
                    "recip_rA",
                    "recip_dxC",
                    "recip_dyC",
                    "recip_hIceMean",
                )
            },
            uIce=u * ones,
            vIce=v * ones,
        )

    return build


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("velocity", [-1, 0, 1])
def test_cfl_one_is_exact_periodic_translation(
    transport, transport_state, sett, axis, velocity
):
    interior = np.random.default_rng(42).uniform(0.2, 2, (5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    vs = transport_state(velocity if axis == 0 else 0, velocity if axis == 1 else 0)
    actual = transport.calc_Advection(vs, sett._replace(deltatTherm=1), field)
    expected = np.roll(interior, velocity, axis=axis)
    np.testing.assert_allclose(actual[2:-2, 2:-2], expected, atol=1e-14)
    assert float(jnp.sum(actual[2:-2, 2:-2])) == pytest.approx(interior.sum())


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("velocity", [-0.7, -0.2, 0.2, 0.7])
def test_subcfl_transport_conserves_mass_and_bounds(
    transport, transport_state, sett, axis, velocity
):
    interior = np.random.default_rng(19).uniform(0.2, 2, (5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    vs = transport_state(velocity if axis == 0 else 0, velocity if axis == 1 else 0)
    actual = np.asarray(
        transport.calc_Advection(vs, sett._replace(deltatTherm=1), field)
    )[2:-2, 2:-2]
    assert actual.sum() == pytest.approx(interior.sum(), abs=1e-12)
    assert actual.min() >= interior.min() - 1e-14
    assert actual.max() <= interior.max() + 1e-14


@pytest.mark.parametrize(
    "ratio, expected",
    [(-2, 0), (0, 0), (0.25, 0.5), (0.5, 1), (1, 1), (1.5, 1.5), (2, 2), (5, 2)],
)
def test_superbee_limiter_breakpoints(transport, ratio, expected):
    assert float(transport.limiter(jnp.asarray(float(ratio)))) == expected
