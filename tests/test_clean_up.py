"""Check post-transport threshold handling without hiding lost overshoots."""

import numpy as np
import pytest

from veris.clean_up import clean_up_advection, ridging


@pytest.mark.parametrize("ice", [-0.2, 0, 0.5, 1, 2])
@pytest.mark.parametrize("snow", [-0.1, 0, 0.3])
def test_cleanup_thresholds_and_overshoots(state, sett, ice, snow):
    sett = sett._replace(hIce_min=0.5, Area_min=0.01)
    vs = state(hIceMean=[[ice]], hSnowMean=[[snow]], Area=[[-0.2]], TSurf=[[260]])
    result = clean_up_advection(vs, sett)
    expected = (0, 0, 0, sett.celsius2K, max(-ice, 0), max(-snow, 0))
    if ice > sett.hIce_min:
        expected = (ice, max(snow, 0), sett.Area_min, 260, 0, max(-snow, 0))
    assert len(result) == len(expected)
    for value, reference in zip(result, expected):
        assert value.shape == (1, 1)
        np.testing.assert_allclose(value, reference)


def test_cleanup_preserves_positive_area_above_minimum(state, sett):
    sett = sett._replace(hIce_min=0.5, Area_min=0.01)
    area = np.array([[0.005, 0.01, 0.4], [0.8, 1.0, 1.2]])
    ice = np.full_like(area, 2.0)
    snow = np.full_like(area, 0.3)
    temperature = np.full_like(area, 260.0)
    result = clean_up_advection(
        state(hIceMean=ice, hSnowMean=snow, Area=area, TSurf=temperature), sett
    )
    expected_area = np.array([[0.01, 0.01, 0.4], [0.8, 1.0, 1.2]])
    expected = (ice, snow, expected_area, temperature, 0, 0)
    assert len(result) == len(expected)
    for value, reference in zip(result, expected):
        assert value.shape == area.shape
        np.testing.assert_array_equal(value, reference)


def test_ridging_caps_only_area(state, sett):
    vs = state(Area=[[-0.1, 0, 0.7, 1, 1.5]])
    result = ridging(vs, sett)
    assert result.shape == vs.Area.shape
    np.testing.assert_array_equal(result, [[-0.1, 0, 0.7, 1, 1]])
