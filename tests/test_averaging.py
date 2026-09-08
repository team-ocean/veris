"""Check coastal corner averaging using explicit four-cell neighborhoods."""

import numpy as np
import pytest
from conftest import StateFactory

from veris.averaging import c_point_to_z_point
from veris.state import Settings


@pytest.mark.parametrize("no_slip", [True, False])
@pytest.mark.parametrize("seed", range(6))
def test_corner_average_with_land(
    state: StateFactory, sett: Settings, no_slip: bool, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, 2, (4, 7))
    field = rng.normal(size=mask.shape) * mask
    result = c_point_to_z_point(
        state(iceMask=mask), sett._replace(noSlip=no_slip), field
    )
    assert result.shape == field.shape
    expected = np.zeros_like(field)
    for i, j in np.ndindex(field.shape):
        cells = [(i, j), (i - 1, j), (i, j - 1), (i - 1, j - 1)]
        count = sum(mask[x, y] for x, y in cells)
        if count and (no_slip or count == 4):
            expected[i, j] = sum(field[x, y] for x, y in cells) / count
    np.testing.assert_allclose(result, expected, atol=1e-14)


@pytest.mark.parametrize("no_slip", [True, False])
@pytest.mark.parametrize("ocean", [False, True])
def test_corner_average_uniform_masks(
    state: StateFactory, sett: Settings, no_slip: bool, ocean: bool
) -> None:
    mask = np.full((3, 5), float(ocean))
    field = np.arange(15, dtype=float).reshape(mask.shape) * mask
    result = c_point_to_z_point(
        state(iceMask=mask), sett._replace(noSlip=no_slip), field
    )
    expected = np.zeros_like(field)
    if ocean:
        for i, j in np.ndindex(field.shape):
            expected[i, j] = (
                field[i, j] + field[i - 1, j] + field[i, j - 1] + field[i - 1, j - 1]
            ) / 4
    assert result.shape == field.shape
    np.testing.assert_array_equal(result, expected)
