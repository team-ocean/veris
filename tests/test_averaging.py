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
    # Every binary four-cell coastline orientation occurs at an odd/odd corner,
    # including the all-land and all-ocean neighborhoods (patterns 0 and 15).
    mask = np.zeros((8, 12), dtype=int)
    for pattern in range(16):
        row, column = 2 * (pattern // 6), 2 * (pattern % 6)
        mask[row : row + 2, column : column + 2] = np.array(
            [(pattern >> bit) & 1 for bit in range(4)]
        ).reshape(2, 2)
    # Integer inputs make four-wet-cell averages exactly representable.
    field = rng.integers(-20, 21, size=mask.shape) * mask
    result = c_point_to_z_point(
        state(iceMask=mask), replace(conf, noSlip=no_slip), phys, field
    )
    assert result.shape == field.shape
    expected = np.zeros_like(field, dtype=float)
    wet_count = np.zeros_like(mask)
    for i, j in np.ndindex(field.shape):
        cells = [(i, j), (i - 1, j), (i, j - 1), (i - 1, j - 1)]
        count = sum(mask[x, y] for x, y in cells)
        wet_count[i, j] = count
        if count and (no_slip or count == 4):
            expected[i, j] = sum(field[x, y] for x, y in cells) / count
    np.testing.assert_allclose(result, expected, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(result)[expected == 0], 0)
    np.testing.assert_array_equal(
        np.asarray(result)[wet_count == 4], expected[wet_count == 4]
    )
