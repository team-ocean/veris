"""Check wind momentum transfer, hydrostatic tilt, and free-drift dispatch."""

import importlib
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest
from conftest import StateFactory

from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants


@pytest.fixture
def dynamics(halo: ModuleType) -> ModuleType:
    """Import dynamics after selecting the standalone periodic halo backend."""
    return importlib.import_module("veris.dynsolver")


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("hemisphere", [-1, 0, 1])
@pytest.mark.parametrize("angle", [0, 30, 90])
def test_wind_stress_rotation_staggering_and_masks(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    dynamics: ModuleType,
    relative: bool,
    hemisphere: int,
    angle: int,
) -> None:
    sett = replace(sett, useRelativeWind=relative)
    phys = replace(phys, airTurnAngle=angle, airIceDrag=0.001, airIceDrag_south=0.002)
    x, y = np.indices((3, 5), dtype=float)
    wind_u, wind_v = 2 + x / 5, -1 + y / 10
    ice_u, ice_v = 0.2 + y / 100, -0.1 + x / 100
    mask_u, mask_v = np.ones_like(x), np.ones_like(x)
    mask_u[0, 1] = 0
    mask_v[1, 0] = 0
    vs = state(
        uWind=wind_u,
        vWind=wind_v,
        uIce=ice_u,
        vIce=ice_v,
        fCori=np.full_like(x, hemisphere * 1e-4),
        iceMaskU=mask_u,
        iceMaskV=mask_v,
    )
    centered_x, centered_y = np.zeros_like(x), np.zeros_like(x)
    coefficient = phys.rhoAir * (
        phys.airIceDrag_south if hemisphere < 0 else phys.airIceDrag
    )
    cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    for i, j in np.ndindex(x.shape):
        u, v = wind_u[i, j], wind_v[i, j]
        if relative:
            u -= (ice_u[i, j] + ice_u[(i + 1) % 3, j]) / 2
            v -= (ice_v[i, j] + ice_v[i, (j + 1) % 5]) / 2
        speed = np.hypot(u, v)
        centered_x[i, j] = coefficient * speed * (cosine * u - hemisphere * sine * v)
        centered_y[i, j] = coefficient * speed * (cosine * v + hemisphere * sine * u)
    expected_x, expected_y = np.zeros_like(x), np.zeros_like(x)
    for i, j in np.ndindex(x.shape):
        expected_x[i, j] = (centered_x[i, j] + centered_x[i - 1, j]) / 2 * mask_u[i, j]
        expected_y[i, j] = (centered_y[i, j] + centered_y[i, j - 1]) / 2 * mask_v[i, j]
    for value, expected in zip(
        dynamics.tauXY(vs, sett, phys), (expected_x, expected_y)
    ):
        assert value.shape == x.shape
        np.testing.assert_allclose(value, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("speed", [0, 0.5, 2])
def test_wind_speed_floor_preserves_zero_stress_at_rest(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    dynamics: ModuleType,
    speed: float,
) -> None:
    sett = replace(sett, useRelativeWind=False, wSpeedMin=1)
    ones = np.ones((3, 5))
    vs = state(
        uWind=speed * ones,
        vWind=0 * ones,
        fCori=ones,
        iceMaskU=ones,
        iceMaskV=ones,
    )
    tx, ty = dynamics.tauXY(vs, sett, phys)
    assert tx.shape == ty.shape == ones.shape
    np.testing.assert_allclose(
        tx, phys.rhoAir * phys.airIceDrag * max(1, speed) * speed
    )
    np.testing.assert_array_equal(ty, 0)


@pytest.mark.parametrize("real_freshwater", [False, True])
@pytest.mark.parametrize("source", ["elevation", "pressure", "load"])
def test_affine_hydrostatic_tilt_and_wind_force(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    dynamics: ModuleType,
    real_freshwater: bool,
    source: str,
) -> None:
    sett = replace(sett, useRelativeWind=False, useRealFreshWaterFlux=real_freshwater)
    x, y = np.indices((4, 7), dtype=float)
    ones = np.ones_like(x)
    dx, dy = 2000, 3000
    slope_x, slope_y = 0.002, -0.003
    field = slope_x * dx * x + slope_y * dy * y
    vs = state(
        uWind=2 * ones,
        vWind=0 * ones,
        fCori=ones,
        iceMaskU=ones,
        iceMaskV=ones,
        AreaW=0.3 * ones,
        AreaS=0.6 * ones,
        ssh_an=field if source == "elevation" else 0 * ones,
        surfPress=100000 + (field if source == "pressure" else 0 * ones),
        SeaIceLoad=900 + (field if source == "load" else 0 * ones),
        SeaIceMassU=800 * ones,
        SeaIceMassV=600 * ones,
        recip_dxC=ones / dx,
        recip_dyC=ones / dy,
    )
    factors = {
        "elevation": phys.gravity,
        "pressure": phys.recip_rhoSea,
        "load": phys.gravity * sett.seaIceLoadFac * phys.recip_rhoSea
        if real_freshwater
        else 0,
    }
    acceleration = factors[source]
    expected = (
        0.3 * phys.rhoAir * phys.airIceDrag * 4 - 800 * acceleration * slope_x,
        -600 * acceleration * slope_y,
    )
    for value, reference in zip(dynamics.WindForcingXY(vs, sett, phys), expected):
        assert value.shape == x.shape
        np.testing.assert_allclose(value[1:, 1:], reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("forcing", [0, 0.05, 0.3])
def test_free_drift_dispatch_preserves_internal_stresses(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    dynamics: ModuleType,
    forcing: float,
) -> None:
    sett = replace(sett, useFreedrift=True, useEVP=False)
    ones = np.ones((3, 5))
    sigma = np.arange(15, dtype=float).reshape(ones.shape)
    vs = state(
        WindForcingX=forcing * ones,
        WindForcingY=0 * ones,
        hIceMean=ones,
        fCori=0 * ones,
        uOcean=0.1 * ones,
        vOcean=-0.2 * ones,
        iceMaskU=ones,
        iceMaskV=ones,
        sigma1=sigma,
        sigma2=-2 * sigma,
        sigma12=0.5 * sigma,
    )
    result = dynamics.IceVelocities(vs, sett, phys)
    # At the equator, quadratic water drag alone balances the applied stress.
    expected_u = 0.1 + np.sqrt(forcing / (phys.rhoSea * phys.waterIceDrag))
    assert len(result) == 5
    for value in result:
        assert value.shape == ones.shape
    np.testing.assert_allclose(result[0], expected_u, rtol=1e-13)
    np.testing.assert_allclose(result[1], -0.2, atol=1e-14)
    for value, expected in zip(result[2:], (sigma, -2 * sigma, 0.5 * sigma)):
        np.testing.assert_array_equal(value, expected)
