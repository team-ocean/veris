"""Verify free drift against uniform momentum balance and land masks."""

import numpy as np
import pytest
from conftest import StateFactory

from veris.configuration import Configuration
from veris.freedrift_solver import freedrift_solver
from veris.physical_constants import PhysicalConstants


@pytest.mark.parametrize("coriolis", [-1e-4, 0, 1e-4])
@pytest.mark.parametrize("wind", [0, 0.05, 0.3])
@pytest.mark.parametrize("ocean_velocity", [(0, 0), (0.15, -0.08), (-0.1, 0.2)])
def test_uniform_free_drift_momentum_balance(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    coriolis: float,
    wind: float,
    ocean_velocity: tuple[float, float],
) -> None:
    ones = np.ones((4, 7))
    vs = state(
        WindForcingX=wind * ones,
        WindForcingY=0.3 * wind * ones,
        hIceMean=ones,
        fCori=coriolis * ones,
        uOcean=ocean_velocity[0] * ones,
        vOcean=ocean_velocity[1] * ones,
        iceMaskU=ones,
        iceMaskV=ones,
    )
    u_jax, v_jax = freedrift_solver(vs, conf, phys)
    u, v = np.asarray(u_jax), np.asarray(v_jax)
    assert u.shape == v.shape == ones.shape
    drag = phys.rhoSea * (
        phys.waterIceDrag_south if coriolis < 0 else phys.waterIceDrag
    )
    relative_u = u - ocean_velocity[0]
    relative_v = v - ocean_velocity[1]
    speed = np.hypot(relative_u, relative_v)
    np.testing.assert_allclose(
        drag * speed * relative_u - phys.rhoIce * coriolis * v, wind, atol=1e-12
    )
    np.testing.assert_allclose(
        drag * speed * relative_v + phys.rhoIce * coriolis * u,
        0.3 * wind,
        atol=1e-12,
    )


def test_free_drift_applies_each_staggered_land_mask(
    state: StateFactory, conf: Configuration, phys: PhysicalConstants
) -> None:
    """Local output masks suppress land without altering neighboring wet faces."""
    ones = np.ones((4, 7))
    mask_u, mask_v = ones.copy(), ones.copy()
    mask_u[1, 2] = 0
    mask_v[2, 3] = 0
    wind = 0.05
    vs = state(
        WindForcingX=wind * ones,
        WindForcingY=0 * ones,
        hIceMean=ones,
        fCori=0 * ones,
        uOcean=0.15 * ones,
        vOcean=-0.08 * ones,
        iceMaskU=mask_u,
        iceMaskV=mask_v,
    )
    expected_u = 0.15 + np.sqrt(wind / (phys.rhoSea * phys.waterIceDrag))
    for component, expected, mask in zip(
        freedrift_solver(vs, conf, phys), (expected_u, -0.08), (mask_u, mask_v)
    ):
        assert component.shape == ones.shape
        np.testing.assert_array_equal(np.asarray(component)[mask == 0], 0)
        np.testing.assert_allclose(
            np.asarray(component)[mask != 0], expected, rtol=1e-13
        )
