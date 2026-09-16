"""Initialize immutable staggered geometry from nonuniform ocean-grid fields."""

from dataclasses import FrozenInstanceError, fields, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from veris.setups.ocean import OceanGeometry, initialize_from_ocean


@pytest.fixture
def ocean_grid() -> OceanGeometry:
    """Frozen geometry with distinguishable surface and subsurface masks."""
    x, y = np.indices((6, 9))
    surface = ((x + y) % 3 != 0).astype(float)
    return OceanGeometry(
        maskT=jnp.asarray(np.stack([np.zeros_like(surface), surface], axis=-1)),
        maskU=jnp.asarray(np.stack([surface, 1 - surface], axis=-1)),
        maskV=jnp.asarray(np.stack([surface, surface[:, ::-1]], axis=-1)),
        ht=jnp.asarray(100 + x + y),
        coriolis_t=jnp.asarray((y - 3) * 1e-5),
        dxt=jnp.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]),
        dyt=jnp.arange(3.0, 12.0),
        dxu=jnp.array([3.0, 4.0, 6.0, 8.0, 12.0, 14.0]),
        dyu=jnp.arange(4.0, 13.0),
        area_t=jnp.asarray(10.0 + x + 2 * y),
        area_u=jnp.asarray(20.0 + x + 2 * y),
        area_v=jnp.asarray(30.0 + x + 2 * y),
    )


def test_initialization_surface_masks_and_reciprocals(
    ocean_grid: OceanGeometry,
) -> None:
    result, conf, _phys = initialize_from_ocean(ocean_grid)
    assert (conf.nx, conf.ny) == (2, 5)
    np.testing.assert_array_equal(result.hIceMean, 0)
    for source, output, interior in (
        ("maskT", "iceMask", "maskInC"),
        ("maskU", "iceMaskU", "maskInU"),
        ("maskV", "iceMaskV", "maskInV"),
    ):
        expected = getattr(ocean_grid, source)[:, :, -1]
        np.testing.assert_array_equal(getattr(result, output), expected)
        np.testing.assert_array_equal(getattr(result, interior), expected)
    for name, input_name, axis in (
        ("dxC", "dxt", 0),
        ("dxV", "dxt", 0),
        ("dxU", "dxu", 0),
        ("dxG", "dxu", 0),
        ("dyC", "dyt", 1),
        ("dyV", "dyt", 1),
        ("dyU", "dyu", 1),
        ("dyG", "dyu", 1),
    ):
        spacing = np.asarray(getattr(ocean_grid, input_name))
        expected = np.broadcast_to(spacing[:, None] if axis == 0 else spacing, (6, 9))
        if hasattr(result, name):
            np.testing.assert_array_equal(getattr(result, name), expected)
        if hasattr(result, "recip_" + name):
            np.testing.assert_allclose(
                getattr(result, "recip_" + name), 1 / expected, rtol=1e-14
            )
    for name, source in (
        ("recip_rA", "area_t"),
        ("recip_rAu", "area_u"),
        ("recip_rAv", "area_v"),
    ):
        np.testing.assert_allclose(
            getattr(result, name), 1 / getattr(ocean_grid, source)
        )
    np.testing.assert_array_equal(result.R_low, ocean_grid.ht)
    np.testing.assert_array_equal(result.fCori, ocean_grid.coriolis_t)
    np.testing.assert_array_equal(result.TSurf, np.full((6, 9), 273))
    area = np.asarray(ocean_grid.area_t)
    expected = np.empty_like(area)
    for i, j in np.ndindex(area.shape):
        expected[i, j] = (
            area[i, j] + area[i - 1, j] + area[i, j - 1] + area[i - 1, j - 1]
        ) / 4
    np.testing.assert_allclose(result.rAz, expected, rtol=1e-14)

    assert len(fields(result)) == 70
    with pytest.raises(FrozenInstanceError):
        setattr(ocean_grid, "ht", jnp.zeros((6, 9)))  # noqa: B010 -- test frozen runtime guard


@pytest.mark.parametrize(
    "name,value,reason",
    [
        ("maskT", jnp.ones((6, 9)), "maskT"),
        ("maskU", jnp.ones((6, 9, 0)), "maskU"),
        ("ht", jnp.ones((6, 8)), "ht"),
        ("dxt", jnp.ones((6, 1)), "dxt"),
        ("dyu", jnp.zeros(9), "dyu"),
        ("area_t", jnp.full((6, 9), -1.0), "area_t"),
        ("dxu", jnp.full(6, jnp.nan), "dxu"),
    ],
)
def test_invalid_geometry_is_rejected(
    ocean_grid: OceanGeometry, name: str, value: object, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        initialize_from_ocean(replace(ocean_grid, **{name: value}))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_external_fields_and_static_overrides(
    ocean_grid: OceanGeometry, dtype: str
) -> None:
    """Allocate forcing, constants and unspecified registry defaults together."""
    from veris.variables import VARIABLES

    temperature = np.full((6, 9), 271.25)
    result, conf, phys = initialize_from_ocean(
        ocean_grid,
        dtype=dtype,
        settings_overrides={
            "nEVPsteps": 3,
            "nx": 20,
            "geometrySurfaceTemperature": 270.0,
        },
        physical_overrides={"rhoIce": 920.0},
        state_overrides={"theta": temperature, "hIceMean": np.ones((6, 9))},
    )
    assert (conf.nx, conf.ny, conf.nEVPsteps) == (2, 5, 3)
    assert conf.dtype == phys.dtype == dtype
    assert phys.rhoIce == 920
    np.testing.assert_array_equal(result.TSurf, 270.0)
    np.testing.assert_array_equal(result.theta, temperature)
    np.testing.assert_array_equal(result.hIceMean, 1)
    np.testing.assert_array_equal(result.hSnowMean, VARIABLES["hSnowMean"].default)
    for field in fields(result):
        assert getattr(result, field.name).dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"state_overrides": {"unknown": np.ones((6, 9))}}, "unknown State"),
        ({"state_overrides": {"theta": np.ones((2, 5))}}, "theta.*shape"),
        ({"physical_overrides": {"dtype": "float32"}}, "physical dtype"),
        ({"dtype": "float16"}, "dtype"),
    ],
)
def test_invalid_initialization_overrides(
    ocean_grid: OceanGeometry, overrides: dict[str, Any], reason: str
) -> None:
    """The ocean entry point retains shared initialization validation."""
    with pytest.raises(ValueError, match=reason):
        initialize_from_ocean(ocean_grid, **overrides)
