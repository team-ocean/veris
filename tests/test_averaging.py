"""Check coastal corner averaging using explicit four-cell neighborhoods."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import StateFactory

from veris.averaging import c_point_to_z_point
from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants


@pytest.mark.parametrize("no_slip", [True, False])
def test_corner_average_with_land(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    no_slip: bool,
) -> None:
    rng = np.random.default_rng(0)
    # Every binary four-cell coastline orientation occurs at an odd/odd corner.
    mask = np.zeros((8, 12), dtype=int)
    for pattern in range(16):
        row, column = 2 * (pattern // 6), 2 * (pattern % 6)
        mask[row : row + 2, column : column + 2] = np.array(
            [(pattern >> bit) & 1 for bit in range(4)]
        ).reshape(2, 2)
    field = rng.normal(size=mask.shape) * mask
    result = c_point_to_z_point(
        state(iceMask=mask), replace(conf, noSlip=no_slip), phys, field
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
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    no_slip: bool,
    ocean: bool,
) -> None:
    mask = np.full((3, 5), float(ocean))
    field = np.arange(15, dtype=float).reshape(mask.shape) * mask
    result = c_point_to_z_point(
        state(iceMask=mask), replace(conf, noSlip=no_slip), phys, field
    )
    expected = np.zeros_like(field)
    if ocean:
        for i, j in np.ndindex(field.shape):
            expected[i, j] = (
                field[i, j] + field[i - 1, j] + field[i, j - 1] + field[i - 1, j - 1]
            ) / 4
    assert result.shape == field.shape
    np.testing.assert_array_equal(result, expected)
