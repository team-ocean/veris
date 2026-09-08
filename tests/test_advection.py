"""Transport tests use exact donor translations and periodic volume budgets."""

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFactory
from jax.typing import ArrayLike

from veris.state import Settings


@pytest.fixture
def transport(halo: ModuleType) -> ModuleType:
    """Import transport after selecting real serial halo exchange."""
    return importlib.import_module("veris.advection")


@pytest.fixture
def transport_state(state: StateFactory) -> Callable[[ArrayLike, ArrayLike], Any]:
    """Unit metric periodic grid with a rectangular 5 by 7 interior.

    Each result inherits StateFactory's dynamic subset of model fields.
    """

    def build(u: ArrayLike, v: ArrayLike) -> Any:
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
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    sett: Settings,
    axis: int,
    velocity: int,
) -> None:
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
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    sett: Settings,
    axis: int,
    velocity: float,
) -> None:
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
def test_superbee_limiter_breakpoints(
    transport: ModuleType, ratio: float, expected: float
) -> None:
    assert float(transport.limiter(jnp.asarray(float(ratio)))) == expected


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("velocity", [-1, 1])
def test_intensive_field_translates_at_unit_cfl(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    sett: Settings,
    axis: int,
    velocity: int,
) -> None:
    """Unit thickness and divergence-free velocity reduce to tracer translation."""
    interior = np.random.default_rng(6).normal(size=(5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    vs = transport_state(velocity if axis == 0 else 0, velocity if axis == 1 else 0)
    result = transport.calc_Advection(
        vs, sett._replace(deltatTherm=1, extensiveFld=False), field
    )
    np.testing.assert_allclose(
        result[2:-2, 2:-2], np.roll(interior, velocity, axis), atol=1e-14
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_intensive_constant_survives_divergent_velocity(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    sett: Settings,
    axis: int,
) -> None:
    """The advective derivative of a spatially constant tracer is zero."""
    vs = transport_state(0, 0)
    velocity = np.pad(
        np.random.default_rng(8).uniform(-0.3, 0.3, (5, 7)), 2, mode="wrap"
    )
    vs = vs._replace(**{("uIce" if axis == 0 else "vIce"): jnp.asarray(velocity)})
    field = jnp.full((9, 11), 2.3)
    result = transport.calc_Advection(
        vs, sett._replace(deltatTherm=1, extensiveFld=False), field
    )
    np.testing.assert_allclose(result, field, atol=1e-14)


def test_intensive_meridional_sweep_uses_updated_zonal_field(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    sett: Settings,
) -> None:
    """Cross-flow cannot change a tracer constant in the cross-flow direction."""
    interior = np.broadcast_to(np.arange(5.0)[:, None], (5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    v = np.broadcast_to(np.linspace(-0.2, 0.2, 7), (5, 7))
    vs = transport_state(1, 0)._replace(vIce=jnp.asarray(np.pad(v, 2, mode="wrap")))
    result = transport.calc_Advection(
        vs, sett._replace(deltatTherm=1, extensiveFld=False), field
    )
    np.testing.assert_allclose(
        result[2:-2, 2:-2], np.roll(interior, 1, axis=0), atol=1e-14
    )
