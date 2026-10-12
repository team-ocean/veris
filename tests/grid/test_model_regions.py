"""Regional initialization contracts, authored before the Task 7 helper."""

from dataclasses import fields
from typing import Any

import jax
import numpy as np
import pytest

from veris import OceanGeometry
from veris.grid.regions import (
    physical_indices,
    regional_forcing,
    regional_geometry,
    regional_state,
)
from veris.model.configuration import Configuration
from veris.model.variables import VARIABLES


@pytest.mark.parametrize("cyclic_y", [False, True])
def test_packed_indices_cross_partitions_and_only_clip_global_walls(
    cyclic_y: bool,
) -> None:
    conf = Configuration(nx=2, ny=3, use_sharding=True, enable_cyclic_y=cyclic_y)
    x, y = physical_indices((slice(6, 12), slice(7, 14)), conf, (4, 6))
    np.testing.assert_array_equal(x, [0, 1, 2, 3, 0, 1])
    np.testing.assert_array_equal(
        y, [1, 2, 3, 4, 5, 0, 1] if cyclic_y else [1, 2, 3, 4, 5, 5, 5]
    )


def test_regional_geometry_preserves_vertical_masks_and_metric_vectors() -> None:
    x, y = np.indices((4, 6))
    masks = np.stack((x + y, 10 + x + y), axis=-1)
    values: dict[str, Any] = {
        "maskT": masks,
        "maskU": masks + 1,
        "maskV": masks + 2,
        "ht": -(100 + x + y),
        "coriolis_t": x + y,
        "dxt": 10 + np.arange(4),
        "dxu": 20 + np.arange(4),
        "dyt": 30 + np.arange(6),
        "dyu": 40 + np.arange(6),
        "area_t": 100 + x + y,
        "area_u": 200 + x + y,
        "area_v": 300 + x + y,
    }
    geometry = OceanGeometry(**values)
    ix, iy = np.array([3, 0, 1, 2, 3, 0]), np.array([1, 2, 3, 4, 5, 5, 5])
    region = regional_geometry(geometry, ix, iy)
    for item in fields(geometry):
        source = values[item.name]
        expected = (
            source[ix]
            if item.name in ("dxt", "dxu")
            else source[iy]
            if item.name in ("dyt", "dyu")
            else source[ix[:, None], iy[None, :]]
        )
        np.testing.assert_array_equal(getattr(region, item.name), expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_legacy_island_serial_face_masks_retain_startup_roll_halos(dtype: str) -> None:
    """Step-zero legacy evidence includes the outermost unrefreshed mask halos."""
    from veris.setups.island import initialize

    state, _, _ = initialize(6, 8, dtype=dtype)
    physical = np.ones((6, 8), dtype=dtype)
    physical[2:4, 3:5] = 0
    mask = np.pad(physical, 2, mode="wrap")
    expected_u = mask * np.roll(mask, 1, axis=0)
    expected_v = mask * np.roll(mask, 1, axis=1)
    for name in ("iceMaskU", "maskInU"):
        np.testing.assert_array_equal(getattr(state, name), expected_u)
    for name in ("iceMaskV", "maskInV"):
        np.testing.assert_array_equal(getattr(state, name), expected_v)


def test_callback_builds_each_addressable_region_once_for_all_state_leaves() -> None:
    mesh = jax.make_mesh((1, 1), ("x", "y"), devices=jax.devices("cpu")[:1])
    conf = Configuration(nx=2, ny=3, use_sharding=True, dtype="float64")
    called = []

    def build(ix: Any, iy: Any) -> dict[str, Any]:
        assert jax.sharding.get_abstract_mesh().empty
        called.append((ix.copy(), iy.copy()))
        return {
            name: np.full((len(ix), len(iy)), number, dtype=conf.dtype)
            for number, name in enumerate(VARIABLES)
        }

    with jax.set_mesh(mesh):
        state = regional_state(mesh, conf, (2, 3), build)
    assert len(called) == 1
    for number, item in enumerate(fields(state)):
        value = getattr(state, item.name)
        assert value.shape == (6, 7) and value.dtype == np.dtype("float64")
        assert value.sharding.mesh == mesh
        assert np.all(np.asarray(value) == number)


@pytest.mark.parametrize("shape", [(), (2,), (4, 6), (2, 4, 6)])
def test_callback_forcing_keeps_time_axis_and_preserves_dtype(
    shape: tuple[int, ...],
) -> None:
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((1, 1), ("x", "y"), devices=jax.devices("cpu")[:1])
    source = np.arange(np.prod(shape)).reshape(shape)
    result = regional_forcing(source, "float64", mesh)
    expected = (
        P(None, "x", "y")
        if len(shape) == 3
        else P("x", "y")
        if len(shape) == 2
        else P()
    )
    assert (
        isinstance(result.sharding, NamedSharding) and result.sharding.spec == expected
    )
    assert result.dtype == np.dtype("float64")
    np.testing.assert_array_equal(result, source)
