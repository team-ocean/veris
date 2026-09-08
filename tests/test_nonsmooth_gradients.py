"""Check branch sensitivities of cleanup, ridging, and the Superbee limiter.

Literal slopes come from the piecewise physical maps. At continuous kinks,
JAX's selected linearization is checked separately from the unequal one-sided
slopes; it is not a classical derivative. Thin-ice removal is discontinuous,
so its branch derivative must not be mistaken for sensitivity across removal.
"""

from collections.abc import Callable, Sequence
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFactory
from jax import Array
from jax.typing import ArrayLike

from veris.clean_up import clean_up_advection, ridging
from veris.state import Settings


def _check_elementwise_ad(
    function: Callable[[Array], Array],
    values: ArrayLike | Sequence[float],
    slopes: ArrayLike | Sequence[float],
) -> None:
    """Check forward/reverse sensitivities against an independent diagonal map."""
    values = jnp.asarray(values, dtype=jnp.float64)
    slopes = np.asarray(slopes)
    tangent = jnp.linspace(-0.7, 1.3, values.size).reshape(values.shape)
    cotangent = jnp.linspace(1.1, -0.4, values.size).reshape(values.shape)
    _, forward = jax.jvp(function, (values,), (tangent,))
    _, pullback = jax.vjp(function, values)
    reverse = pullback(cotangent)[0]
    np.testing.assert_allclose(forward, slopes * tangent, atol=1e-14)
    np.testing.assert_allclose(reverse, slopes * cotangent, atol=1e-14)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, forward), jnp.vdot(reverse, tangent), atol=1e-14
    )


def test_superbee_piecewise_slopes_and_selected_kink_linearizations(
    halo: ModuleType,
) -> None:
    """Catch wrong limiter branches and unintended AD changes at their joins."""
    from veris.advection import limiter

    points = [-0.2, 0.0, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 2.4]
    expected = [0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0]
    # Nested max ties select 3/4 at zero; these are algorithmic choices.
    slopes = [0.0, 0.75, 2.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.0]
    np.testing.assert_allclose(limiter(jnp.asarray(points)), expected)
    _check_elementwise_ad(limiter, points, slopes)


@pytest.mark.parametrize(
    ("point", "left_slope", "right_slope"),
    [(0.0, 0.0, 2.0), (0.5, 2.0, 0.0), (1.0, 0.0, 1.0), (2.0, 1.0, 0.0)],
)
def test_superbee_one_sided_slopes_at_kinks(
    halo: ModuleType, point: float, left_slope: float, right_slope: float
) -> None:
    """The limiter has distinct one-sided slopes, not a central derivative."""
    from veris.advection import limiter

    step = 1e-6
    center = limiter(jnp.asarray(point))
    left = (center - limiter(jnp.asarray(point - step))) / step
    right = (limiter(jnp.asarray(point + step)) - center) / step
    np.testing.assert_allclose([left, right], [left_slope, right_slope], atol=2e-10)


def test_ridging_area_cap_ad_and_one_sided_slopes(
    state: StateFactory, sett: Settings
) -> None:
    """Saturated area must have zero sensitivity; the cap itself is a kink."""

    def capped(area: Array) -> Array:
        return ridging(state(Area=area), sett)

    _check_elementwise_ad(capped, [0.4, 1.0, 1.3], [1.0, 0.5, 0.0])
    step = 1e-6
    left = (capped(jnp.asarray(1.0)) - capped(jnp.asarray(1.0 - step))) / step
    right = (capped(jnp.asarray(1.0 + step)) - capped(jnp.asarray(1.0))) / step
    np.testing.assert_allclose([left, right], [1.0, 0.0], atol=1e-10)


@pytest.mark.parametrize("output_index", [1, 5], ids=["snow", "snow_overshoot"])
def test_cleanup_snow_zero_preserves_overshoot_sensitivities(
    state: StateFactory, sett: Settings, output_index: int
) -> None:
    """Clipping transfers negative snow to overshoot, retaining its sign in AD."""

    def cleaned(snow: Array) -> Array:
        vs = state(hIceMean=jnp.ones_like(snow), hSnowMean=snow, Area=0.5, TSurf=260)
        return clean_up_advection(vs, sett)[output_index]

    slopes = [0.0, 0.5, 1.0] if output_index == 1 else [-1.0, -0.5, 0.0]
    _check_elementwise_ad(cleaned, [-0.2, 0.0, 0.2], slopes)
    step = 1e-6
    left = (cleaned(jnp.asarray(0.0)) - cleaned(jnp.asarray(-step))) / step
    right = (cleaned(jnp.asarray(step)) - cleaned(jnp.asarray(0.0))) / step
    expected = [0.0, 1.0] if output_index == 1 else [-1.0, 0.0]
    np.testing.assert_allclose([left, right], expected, atol=1e-14)


def test_cleanup_area_floor_ad_and_one_sided_slopes(
    state: StateFactory, sett: Settings
) -> None:
    """The area floor suppresses area sensitivity while positive ice remains."""
    sett = sett._replace(Area_min=0.1)

    def cleaned(area: Array) -> Array:
        vs = state(hIceMean=1.0, hSnowMean=0.2, Area=area, TSurf=260.0)
        return clean_up_advection(vs, sett)[2]

    _check_elementwise_ad(cleaned, [-0.1, 0.0, 0.05, 0.1, 0.3], [0, 0, 0, 0.5, 1])
    step = 1e-6
    left = (cleaned(jnp.asarray(0.1)) - cleaned(jnp.asarray(0.1 - step))) / step
    right = (cleaned(jnp.asarray(0.1 + step)) - cleaned(jnp.asarray(0.1))) / step
    np.testing.assert_allclose([left, right], [0.0, 1.0], atol=1e-10)


def test_thin_ice_removal_branch_ad_does_not_describe_discontinuous_jump(
    state: StateFactory, sett: Settings
) -> None:
    """Catch a changed removal boundary and expose the jump that AD cannot see."""
    sett = sett._replace(hIce_min=0.1)

    def cleaned(ice: Array) -> Array:
        vs = state(hIceMean=ice, hSnowMean=0.2, Area=0.5, TSurf=260.0)
        return clean_up_advection(vs, sett)[0]

    _check_elementwise_ad(cleaned, [0.05, 0.1, 0.15], [0.0, 0.0, 1.0])
    # <= removes ice exactly at the threshold. The jump remains O(hIce_min)
    # as the perturbation shrinks; no finite derivative spans that boundary.
    steps = jnp.asarray([1e-3, 1e-5, 1e-7])
    jumps = cleaned(sett.hIce_min + steps) - cleaned(jnp.asarray(sett.hIce_min))
    np.testing.assert_allclose(jumps, [0.101, 0.10001, 0.1000001], atol=1e-14)
    np.testing.assert_array_equal(cleaned(sett.hIce_min - steps), 0.0)
