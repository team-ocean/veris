"""Initialize staggered geometry from explicit nonuniform ocean-grid fields."""

import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def ocean_grid():
    """Mutable host state with distinguishable surface and subsurface masks."""
    x, y = np.indices((4, 7))
    surface = ((x + y) % 3 != 0).astype(float)
    vs = SimpleNamespace(
        maskT=jnp.asarray(np.stack([np.zeros_like(surface), surface], axis=-1)),
        maskU=jnp.asarray(np.stack([surface, 1 - surface], axis=-1)),
        maskV=jnp.asarray(np.stack([surface, surface[:, ::-1]], axis=-1)),
        ht=jnp.asarray(100 + x + y),
        coriolis_t=jnp.asarray((y - 3) * 1e-5),
        dxt=jnp.array([2.0, 3.0, 5.0, 7.0]),
        dyt=jnp.arange(3.0, 10.0),
        dxu=jnp.array([3.0, 4.0, 6.0, 8.0]),
        dyu=jnp.arange(4.0, 11.0),
        area_t=jnp.asarray(10.0 + x + 2 * y),
        area_u=jnp.asarray(20.0 + x + 2 * y),
        area_v=jnp.asarray(30.0 + x + 2 * y),
    )
    return SimpleNamespace(variables=vs)


def test_initialization_surface_masks_and_reciprocals(ocean_grid):
    initialize = importlib.import_module("veris.set_inits").set_inits
    initialize(ocean_grid)
    vs = ocean_grid.variables
    for source, output in (
        ("maskT", "iceMask"),
        ("maskU", "iceMaskU"),
        ("maskV", "iceMaskV"),
    ):
        np.testing.assert_array_equal(
            getattr(vs, output), getattr(vs, source)[:, :, -1]
        )
    for mask, interior in (
        ("iceMask", "maskInC"),
        ("iceMaskU", "maskInU"),
        ("iceMaskV", "maskInV"),
    ):
        np.testing.assert_array_equal(getattr(vs, mask), getattr(vs, interior))
    for name in (
        "dxC",
        "dyC",
        "dxG",
        "dyG",
        "dxU",
        "dyU",
        "dxV",
        "dyV",
        "rA",
        "rAu",
        "rAv",
        "rAz",
    ):
        actual = getattr(vs, name)
        assert actual.shape == (4, 7)
        np.testing.assert_allclose(actual * getattr(vs, "recip_" + name), 1, rtol=1e-14)
    for name, input_name in (
        ("dxC", "dxt"),
        ("dxV", "dxt"),
        ("dxU", "dxu"),
        ("dxG", "dxu"),
    ):
        expected = np.broadcast_to(np.asarray(getattr(vs, input_name))[:, None], (4, 7))
        np.testing.assert_array_equal(getattr(vs, name), expected)
    for name, input_name in (
        ("dyC", "dyt"),
        ("dyV", "dyt"),
        ("dyU", "dyu"),
        ("dyG", "dyu"),
    ):
        expected = np.broadcast_to(np.asarray(getattr(vs, input_name)), (4, 7))
        np.testing.assert_array_equal(getattr(vs, name), expected)
    np.testing.assert_array_equal(vs.R_low, vs.ht)
    np.testing.assert_array_equal(vs.fCori, vs.coriolis_t)
    np.testing.assert_array_equal(vs.TSurf, np.full((4, 7), 273))


def test_corner_area_is_four_cell_mean(ocean_grid):
    initialize = importlib.import_module("veris.set_inits").set_inits
    initialize(ocean_grid)
    vs = ocean_grid.variables
    area = np.asarray(vs.area_t)
    expected = np.empty_like(area)
    for i, j in np.ndindex(area.shape):
        expected[i, j] = (
            area[i, j] + area[i - 1, j] + area[i, j - 1] + area[i - 1, j - 1]
        ) / 4
    np.testing.assert_allclose(vs.rAz, expected, rtol=1e-14)


def test_uniform_grid_preserves_cell_area(ocean_grid):
    vs = ocean_grid.variables
    vs.area_t = jnp.full((4, 7), 12.0)
    importlib.import_module("veris.set_inits").set_inits(ocean_grid)
    np.testing.assert_array_equal(vs.rAz, np.full((4, 7), 12.0))
