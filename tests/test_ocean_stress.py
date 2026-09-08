"""Ice-ocean stresses must oppose relative motion and rotate by turning angle."""

import importlib
from types import ModuleType

import numpy as np
import pytest
from conftest import StateFactory

from veris.state import Settings


@pytest.mark.parametrize("hemisphere", [-1, 1])
@pytest.mark.parametrize("angle", [0, 25])
@pytest.mark.parametrize("relative", [(0, 0), (0.2, -0.1)])
def test_uniform_ocean_stress_rotation(
    halo: ModuleType,
    state: StateFactory,
    sett: Settings,
    hemisphere: int,
    angle: int,
    relative: tuple[float, float],
) -> None:
    ocean = importlib.import_module("veris.ocean_stress")
    sett = sett._replace(waterTurnAngle=angle, waterIceDrag_south=0.007)
    ones = np.ones((8, 11))
    du, dv = relative
    vs = state(
        uIce=(0.1 + du) * ones,
        vIce=(-0.3 + dv) * ones,
        uOcean=0.1 * ones,
        vOcean=-0.3 * ones,
        maskInU=ones,
        maskInV=ones,
        iceMask=ones,
        fCori=hemisphere * ones,
    )
    drag = max(
        sett.cDragMin,
        sett.rhoSea
        * np.hypot(du, dv)
        * (sett.waterIceDrag_south if hemisphere < 0 else sett.waterIceDrag),
    )
    radians = np.deg2rad(angle)
    expected = (
        drag * (np.cos(radians) * du - hemisphere * np.sin(radians) * dv),
        drag * (np.cos(radians) * dv + hemisphere * np.sin(radians) * du),
    )
    for actual, scalar in zip(ocean.OceanStressUV(vs, sett), expected):
        assert actual.shape == ones.shape
        np.testing.assert_allclose(actual, scalar, atol=1e-14)
