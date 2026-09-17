"""Transport tests use exact donor translations and periodic volume budgets."""

import importlib
from collections.abc import Callable
from dataclasses import replace
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFactory
from jax.typing import ArrayLike

from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants


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


@pytest.mark.parametrize("axis,velocity", [(0, -1), (0, 0), (0, 1), (1, -1), (1, 1)])
def test_cfl_one_is_exact_periodic_translation(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    conf: Configuration,
    phys: PhysicalConstants,
    axis: int,
    velocity: int,
) -> None:
    interior = np.random.default_rng(42).uniform(0.2, 2, (5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    vs = transport_state(velocity if axis == 0 else 0, velocity if axis == 1 else 0)
    actual = transport.calc_Advection(vs, replace(conf, deltatTherm=1), phys, field)
    expected = np.roll(interior, velocity, axis=axis)
    np.testing.assert_allclose(actual[2:-2, 2:-2], expected, atol=1e-14)
    assert float(jnp.sum(actual[2:-2, 2:-2])) == pytest.approx(interior.sum())


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("velocity", [-0.7, -0.2, 0.2, 0.7])
def test_subcfl_transport_conserves_mass_and_bounds(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    conf: Configuration,
    phys: PhysicalConstants,
    axis: int,
    velocity: float,
) -> None:
    interior = np.random.default_rng(19).uniform(0.2, 2, (5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    vs = transport_state(velocity if axis == 0 else 0, velocity if axis == 1 else 0)
    actual = np.asarray(
        transport.calc_Advection(vs, replace(conf, deltatTherm=1), phys, field)
    )[2:-2, 2:-2]
    assert actual.sum() == pytest.approx(interior.sum(), abs=1e-12)
    assert actual.min() >= interior.min() - 1e-14
    assert actual.max() <= interior.max() + 1e-14


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("velocity", [-1, 1])
def test_intensive_field_translates_at_unit_cfl(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    conf: Configuration,
    phys: PhysicalConstants,
    axis: int,
    velocity: int,
) -> None:
    """Unit thickness and divergence-free velocity reduce to tracer translation."""
    interior = np.random.default_rng(6).normal(size=(5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    vs = transport_state(velocity if axis == 0 else 0, velocity if axis == 1 else 0)
    result = transport.calc_Advection(
        vs, replace(conf, deltatTherm=1, extensiveFld=False), phys, field
    )
    np.testing.assert_allclose(
        result[2:-2, 2:-2], np.roll(interior, velocity, axis), atol=1e-14
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_intensive_constant_survives_divergent_velocity(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    conf: Configuration,
    phys: PhysicalConstants,
    axis: int,
) -> None:
    """The advective derivative of a spatially constant tracer is zero."""
    vs = transport_state(0, 0)
    velocity = np.pad(
        np.random.default_rng(8).uniform(-0.3, 0.3, (5, 7)), 2, mode="wrap"
    )
    vs = replace(vs, **{"uIce" if axis == 0 else "vIce": jnp.asarray(velocity)})
    field = jnp.full((9, 11), 2.3)
    result = transport.calc_Advection(
        vs, replace(conf, deltatTherm=1, extensiveFld=False), phys, field
    )
    np.testing.assert_allclose(result, field, atol=1e-14)


def test_intensive_meridional_sweep_uses_updated_zonal_field(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    conf: Configuration,
    phys: PhysicalConstants,
) -> None:
    """Cross-flow cannot change a tracer constant in the cross-flow direction."""
    interior = np.broadcast_to(np.arange(5.0)[:, None], (5, 7))
    field = jnp.asarray(np.pad(interior, 2, mode="wrap"))
    v = np.broadcast_to(np.linspace(-0.2, 0.2, 7), (5, 7))
    vs = replace(transport_state(1, 0), vIce=jnp.asarray(np.pad(v, 2, mode="wrap")))
    result = transport.calc_Advection(
        vs, replace(conf, deltatTherm=1, extensiveFld=False), phys, field
    )
    np.testing.assert_allclose(
        result[2:-2, 2:-2], np.roll(interior, 1, axis=0), atol=1e-14
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("cyclic_y", [False, True])
def test_directional_flux_matches_scalar_stencil(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    conf: Configuration,
    phys: PhysicalConstants,
    axis: int,
    cyclic_y: bool,
) -> None:
    """Check limiter branches, staggered metrics, masks and complete wall halos."""
    rng = np.random.default_rng(918)
    shape = (9, 11)
    field = rng.uniform(-1, 2, shape)
    field[3:6, 3:6] = 0.75  # Exercise exact zero denominators and zero slopes.
    velocity = rng.uniform(-0.7, 0.7, shape)
    reciprocal = rng.uniform(0.4, 1.4, shape)
    face_mask = rng.choice([0.0, 0.4, 1.0], shape)
    interior_mask = rng.choice([0.0, 0.7, 1.0], shape)
    transport_values = velocity * rng.uniform(0.3, 1.1, shape)
    names = (
        ("uIce", "recip_dxC", "iceMaskU", "maskInU")
        if axis == 0
        else ("vIce", "recip_dyC", "iceMaskV", "maskInV")
    )
    vs = replace(
        transport_state(0, 0),
        **dict(
            zip(
                names,
                map(jnp.asarray, (velocity, reciprocal, face_mask, interior_mask)),
                strict=True,
            )
        ),
    )
    settings = replace(conf, deltatTherm=0.7, CrMax=0.6, enable_cyclic_y=cyclic_y)
    raw = np.zeros(shape)
    for index in np.ndindex(shape):
        if not 2 <= index[axis] < shape[axis] - 1:
            continue

        def shifted(offset: int, index: tuple[int, int] = index) -> tuple[int, int]:
            return (
                index[0] + (offset if axis == 0 else 0),
                index[1] + (offset if axis == 1 else 0),
            )

        slopes = [
            (field[shifted(offset)] - field[shifted(offset - 1)])
            * face_mask[shifted(offset)]
            * interior_mask[shifted(offset)]
            for offset in (-1, 0, 1)
        ]
        slope = slopes[1]
        upstream = slopes[0] if transport_values[index] > 0 else slopes[2]
        ratio = (
            upstream / slope
            if abs(slope) * settings.CrMax > abs(upstream)
            else np.sign(upstream) * settings.CrMax * np.sign(slope)
        )
        limited = max(0, min(1, 2 * ratio), min(2, ratio))
        cfl = abs(velocity[index] * settings.deltatTherm * reciprocal[index])
        raw[index] = (
            transport_values[index] * (field[index] + field[shifted(-1)]) * 0.5
            - abs(transport_values[index])
            * ((1 - limited) + cfl * limited)
            * slope
            * 0.5
        )

    expected = np.zeros(shape)
    for i, j in np.ndindex(shape):
        source_i = 2 + (i - 2) % (shape[0] - 4)
        if cyclic_y:
            source_j = 2 + (j - 2) % (shape[1] - 4)
        elif axis == 1 and (j <= 2 or j >= shape[1] - 2):
            continue
        else:
            source_j = min(max(j, 2), shape[1] - 3)
        expected[i, j] = raw[source_i, source_j]
    function = transport.calc_ZonalFlux if axis == 0 else transport.calc_MeridionalFlux
    actual = function(
        vs, settings, phys, jnp.asarray(field), jnp.asarray(transport_values)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("axis", [0, 1])
def test_directional_flux_weighted_gradient(
    transport: ModuleType,
    transport_state: Callable[[ArrayLike, ArrayLike], Any],
    conf: Configuration,
    phys: PhysicalConstants,
    axis: int,
) -> None:
    """A spatially weighted objective checks local transport sensitivities."""
    rng = np.random.default_rng(513)
    field = jnp.asarray(rng.uniform(0.1, 2.0, (9, 11)))
    perturbation = jnp.asarray(rng.normal(size=(9, 11)))
    weights = jnp.asarray(rng.normal(size=(5, 7)))
    vs = transport_state(0.3, -0.2)
    settings = replace(conf, deltatTherm=0.7)
    function = transport.calc_ZonalFlux if axis == 0 else transport.calc_MeridionalFlux
    velocity = vs.uIce if axis == 0 else vs.vIce

    def objective(amount: jax.Array) -> jax.Array:
        flux = function(vs, settings, phys, field + amount * perturbation, velocity)
        return jnp.sum(weights * flux[2:-2, 2:-2])

    point = jnp.asarray(0.0)
    reverse = jax.grad(objective)(point)
    _, forward = jax.jvp(objective, (point,), (jnp.asarray(1.0),))
    epsilon = 1e-6
    finite_difference = (objective(point + epsilon) - objective(point - epsilon)) / (
        2 * epsilon
    )
    assert abs(float(finite_difference)) > 1e-4
    np.testing.assert_allclose(reverse, forward, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(reverse, finite_difference, rtol=1e-7, atol=1e-9)
