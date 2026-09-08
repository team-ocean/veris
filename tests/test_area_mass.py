"""Verify staggered averages against explicit periodic indexing."""

import numpy as np
import pytest

from veris.area_mass import AreaWS, SeaIceMass


@pytest.mark.parametrize("shape", [(3, 5), (6, 4), (1, 3)])
@pytest.mark.parametrize("seed", range(4))
def test_area_and_mass_staggering(state, sett, shape, seed):
    rng = np.random.default_rng(seed)
    area, ice, snow = rng.random((3, *shape))
    vs = state(Area=area, hIceMean=ice, hSnowMean=snow)
    west, south = AreaWS(vs, sett)
    mass, mass_u, mass_v = SeaIceMass(vs, sett)
    expected = sett.rhoIce * ice + sett.rhoSnow * snow
    np.testing.assert_allclose(mass, expected, rtol=1e-14)
    for i, j in np.ndindex(shape):
        assert west[i, j] == pytest.approx((area[i, j] + area[i - 1, j]) / 2)
        assert south[i, j] == pytest.approx((area[i, j] + area[i, j - 1]) / 2)
        assert mass_u[i, j] == pytest.approx((expected[i, j] + expected[i - 1, j]) / 2)
        assert mass_v[i, j] == pytest.approx((expected[i, j] + expected[i, j - 1]) / 2)
    assert np.sum(mass_u) == pytest.approx(np.sum(expected))
    assert np.sum(mass_v) == pytest.approx(np.sum(expected))
