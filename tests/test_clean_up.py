"""Check post-transport threshold handling without hiding lost overshoots."""

from dataclasses import replace

import numpy as np
from conftest import StateFactory

from veris.clean_up import clean_up_advection, ridging
from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants


def test_cleanup_thresholds_and_overshoots(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
) -> None:
    """Batch all ice/snow threshold combinations through the pointwise kernel."""
    phys = replace(phys, hIce_min=0.5, Area_min=0.01)
    ice, snow = np.meshgrid([-0.2, 0, 0.5, 1, 2], [-0.1, 0, 0.3], indexing="ij")
    vs = state(
        hIceMean=ice,
        hSnowMean=snow,
        Area=np.full_like(ice, -0.2),
        TSurf=np.full_like(ice, 260),
    )
    result = clean_up_advection(vs, conf, phys)
    expected = [np.empty_like(ice) for _ in range(6)]
    for index in np.ndindex(ice.shape):
        height, snowfall = ice[index], snow[index]
        reference = (0, 0, 0, phys.celsius2K, max(-height, 0), max(-snowfall, 0))
        if height > phys.hIce_min:
            reference = (
                height,
                max(snowfall, 0),
                phys.Area_min,
                260,
                0,
                max(-snowfall, 0),
            )
        for field, value in zip(expected, reference, strict=True):
            field[index] = value
    assert len(result) == len(expected)
    for value, reference in zip(result, expected, strict=True):
        assert value.shape == ice.shape
        np.testing.assert_allclose(value, reference)


def test_cleanup_preserves_positive_area_above_minimum(
    state: StateFactory, conf: Configuration, phys: PhysicalConstants
) -> None:
    phys = replace(phys, hIce_min=0.5, Area_min=0.01)
    area = np.array([[0.005, 0.01, 0.4], [0.8, 1.0, 1.2]])
    ice = np.full_like(area, 2.0)
    snow = np.full_like(area, 0.3)
    temperature = np.full_like(area, 260.0)
    result = clean_up_advection(
        state(hIceMean=ice, hSnowMean=snow, Area=area, TSurf=temperature), conf, phys
    )
    expected_area = np.array([[0.01, 0.01, 0.4], [0.8, 1.0, 1.2]])
    expected = (ice, snow, expected_area, temperature, 0, 0)
    assert len(result) == len(expected)
    for value, reference in zip(result, expected):
        assert value.shape == area.shape
        np.testing.assert_array_equal(value, reference)


def test_ridging_caps_only_area(
    state: StateFactory, conf: Configuration, phys: PhysicalConstants
) -> None:
    vs = state(Area=[[-0.1, 0, 0.7, 1, 1.5]])
    result = ridging(vs, conf, phys)
    assert result.shape == vs.Area.shape
    np.testing.assert_array_equal(result, [[-0.1, 0, 0.7, 1, 1]])
