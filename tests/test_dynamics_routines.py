"""Validate drag and Hibler rheology with scalar balances and affine fields."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import StateFactory

from veris.configuration import Settings
from veris.dynamics_routines import (
    SeaIceStrength,
    basal_drag_coeffs,
    ocean_drag_coeffs,
    side_drag,
    strainrates,
    stress,
    stressdiv,
    viscosities,
)
from veris.physical_constants import PhysicalConstants


def test_ice_strength_concentration_and_land(
    state: StateFactory, sett: Settings, phys: PhysicalConstants
) -> None:
    ice = np.array([[0, 1, 2], [3, 1, 2]], dtype=float)
    area = np.array([[0, 0.5, 1], [0.8, 1, 0.9]])
    mask = np.array([[1, 1, 1], [1, 0, 0]])
    result = SeaIceStrength(state(hIceMean=ice, Area=area, iceMask=mask), sett, phys)
    expected = np.zeros_like(ice)
    for i, j in np.ndindex(ice.shape):
        if mask[i, j]:
            expected[i, j] = (
                phys.pStar * ice[i, j] * np.exp(-phys.cStar * (1 - area[i, j]))
            )
    assert result.shape == ice.shape
    np.testing.assert_allclose(result, expected, rtol=1e-13)


@pytest.mark.parametrize("coriolis", [-1e-4, 0, 1e-4])
@pytest.mark.parametrize("velocity", [(0, 0), (0.03, 0.04), (0.3, -0.4)])
def test_ocean_drag_relative_speed_floor_and_land(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    coriolis: float,
    velocity: tuple[float, float],
) -> None:
    phys = replace(phys, waterIceDrag=0.004, waterIceDrag_south=0.008)
    ones = np.ones((3, 5))
    mask = ones.copy()
    mask[1, 2] = 0
    vs = state(
        fCori=coriolis * ones,
        uOcean=0.2 * ones,
        vOcean=-0.1 * ones,
        maskInU=ones,
        maskInV=ones,
        iceMask=mask,
    )
    result = ocean_drag_coeffs(
        vs, sett, phys, (0.2 + velocity[0]) * ones, (-0.1 + velocity[1]) * ones
    )
    coefficient = phys.waterIceDrag_south if coriolis < 0 else phys.waterIceDrag
    expected = max(sett.cDragMin, phys.rhoSea * coefficient * np.hypot(*velocity))
    assert result.shape == ones.shape
    np.testing.assert_allclose(result, expected * mask, rtol=1e-13)


@pytest.mark.parametrize("area", [0, 0.01, 0.02, 0.8, 1])
@pytest.mark.parametrize("velocity", [(0, 0), (0.3, -0.4)])
def test_basal_drag_regularized_keel_threshold(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    area: float,
    velocity: tuple[float, float],
) -> None:
    phys = replace(phys, basalDragK2=0.7)
    ones = np.ones((3, 5))
    vs = state(
        Area=area * ones,
        hIceMean=2 * ones,
        R_low=-16 * ones,
        maskInU=ones,
        maskInV=ones,
    )
    result = basal_drag_coeffs(vs, sett, phys, velocity[0] * ones, velocity[1] * ones)
    expected = 0
    if area > 0.01:
        speed = np.sqrt(0.5 * np.hypot(*velocity) ** 2 + phys.basalDragU0**2)
        keel_excess = 2 - 16 * area / phys.basalDragK1
        expected = (
            phys.basalDragK2
            / speed
            * np.logaddexp(0, 10 * keel_excess)
            / 10
            * np.exp(-phys.cBasalStar * (1 - area))
        )
    assert result.shape == ones.shape
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("coastline", [False, True])
@pytest.mark.parametrize("velocity", [(0, 0), (0.3, 0.4)])
def test_side_drag_coastline_and_neighbor_counts(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    coastline: bool,
    velocity: tuple[float, float],
) -> None:
    sett = replace(sett, use_coastline=coastline)
    ones = np.ones((4, 7))
    mask_u, mask_v = ones.copy(), ones.copy()
    mask_u[0, :] = 0
    mask_v[:, 0] = 0
    vs = state(
        AreaW=ones,
        AreaS=ones,
        iceMaskU=mask_u,
        iceMaskV=mask_v,
        SeaIceMassU=900 * ones,
        SeaIceMassV=600 * ones,
        Fu=0.75 * ones,
        Fv=1.25 * ones,
    )
    result = side_drag(vs, sett, phys, velocity[0] * ones, velocity[1] * ones)
    expected_u, expected_v = np.zeros_like(ones), np.zeros_like(ones)
    for i, j in np.ndindex(ones.shape):
        neighbors_u = mask_u[i, j] * (2 - mask_u[i - 1, j] - mask_u[(i + 1) % 4, j])
        neighbors_v = mask_v[i, j] * (2 - mask_v[i, j - 1] - mask_v[i, (j + 1) % 7])
        factor_u, factor_v = (0.75, 1.25) if coastline else (neighbors_u, neighbors_v)
        denominator = np.hypot(*velocity) + phys.sideDragU0
        expected_u[i, j] = 900 * phys.sideDragCoeff * factor_u / denominator
        expected_v[i, j] = 600 * phys.sideDragCoeff * factor_v / denominator
    for value, reference in zip(result, (expected_u, expected_v)):
        assert value.shape == ones.shape
        np.testing.assert_allclose(value, reference, rtol=1e-13)


@pytest.mark.parametrize("boundary", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("coefficients", [(0, 0, 0, 0), (2, 3, 4, -1), (0, -2, 2, 0)])
def test_affine_cartesian_strain_tensor(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    boundary: tuple[bool, bool],
    coefficients: tuple[int, int, int, int],
) -> None:
    sett = replace(sett, noSlip=boundary[0], secondOrderBC=boundary[1])
    x, y = np.indices((6, 8), dtype=float)
    ones = np.ones_like(x)
    a, b, c, d = coefficients
    vs = state(
        recip_dxU=ones,
        recip_dyV=ones,
        recip_dyU=ones,
        recip_dxV=ones,
        k1AtC=0 * ones,
        k2AtC=0 * ones,
        k1AtZ=0 * ones,
        k2AtZ=0 * ones,
        maskInC=ones,
        iceMask=ones,
        iceMaskU=ones,
        iceMaskV=ones,
    )
    result = strainrates(vs, sett, phys, a * x + b * y + 0.2, c * x + d * y - 0.3)
    for value, reference in zip(result, (a, d, 0.5 * (b + c))):
        assert value.shape == x.shape
        np.testing.assert_allclose(value[1:-1, 1:-1], reference, atol=1e-14)


@pytest.mark.parametrize("strain", [(0, 0, 0), (0.01, 0.01, 0), (0.02, -0.01, 0.03)])
@pytest.mark.parametrize("replacement", [0, 1])
@pytest.mark.parametrize("tensile", [0, 0.2])
def test_uniform_viscosity_and_stress_scalar_equations(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    strain: tuple[float, float, float],
    replacement: int,
    tensile: float,
) -> None:
    sett = replace(sett, pressReplFac=replacement)
    phys = replace(phys, tensileStrFac=tensile)
    ones = np.ones((3, 5))
    strength = 1500.0
    vs = state(rAz=ones, recip_rA=ones, SeaIceStrength=strength * ones, iceMask=ones)
    e11, e22, e12 = strain
    delta = np.sqrt(
        (e11 + e22) ** 2 + ((e11 - e22) ** 2 + 4 * e12**2) / phys.PlasDefCoeff**2
    )
    bulk = strength * (1 + tensile) / (2 * (delta + sett.deltaMin))
    shear = bulk / phys.PlasDefCoeff**2
    pressure = (
        strength
        * (1 - tensile)
        * (1 - replacement + replacement * delta / (delta + sett.deltaMin))
    )
    result = viscosities(vs, sett, phys, e11 * ones, e22 * ones, e12 * ones)
    for value, reference in zip(result, (bulk, shear, pressure)):
        assert value.shape == ones.shape
        np.testing.assert_allclose(value, reference, rtol=1e-13)
    tensor = stress(vs, sett, phys, e11 * ones, e22 * ones, e12 * ones, *result)
    expected = (
        bulk * (e11 + e22) + shear * (e11 - e22) - pressure / 2,
        bulk * (e11 + e22) - shear * (e11 - e22) - pressure / 2,
        2 * shear * e12,
    )
    for value, reference in zip(tensor, expected):
        assert value.shape == ones.shape
        np.testing.assert_allclose(value, reference, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("constant", [False, True])
def test_stress_divergence_affine_cartesian_tensor(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, constant: bool
) -> None:
    x, y = np.indices((6, 8), dtype=float)
    dx, dy = 2.0, 3.0
    ones = np.ones_like(x)
    vs = state(
        dyV=dy * ones,
        dxV=dx * ones,
        dxU=dx * ones,
        dyU=dy * ones,
        recip_rAu=ones / (dx * dy),
        recip_rAv=ones / (dx * dy),
    )
    if constant:
        tensor = (2 * ones, -3 * ones, 0.5 * ones)
        expected = (0, 0)
    else:
        tensor = (2 * dx * x, -3 * dy * y, 0.5 * dx * x + 4 * dy * y)
        expected = (6, -2.5)
    for value, reference in zip(stressdiv(vs, sett, phys, *tensor), expected):
        assert value.shape == x.shape
        region = value if constant else value[1:-1, 1:-1]
        np.testing.assert_allclose(region, reference, atol=1e-13)
