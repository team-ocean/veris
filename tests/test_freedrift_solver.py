"""Verify free drift against uniform momentum balance and land masks."""

import numpy as np
import pytest

from veris.freedrift_solver import freedrift_solver


@pytest.mark.parametrize("coriolis", [-1e-4, 0, 1e-4])
@pytest.mark.parametrize("wind", [0, 0.05, 0.3])
@pytest.mark.parametrize("ocean_velocity", [(0, 0), (0.15, -0.08), (-0.1, 0.2)])
def test_uniform_free_drift_momentum_balance(
    state, sett, coriolis, wind, ocean_velocity
):
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
    u_jax, v_jax = freedrift_solver(vs, sett)
    u, v = np.asarray(u_jax), np.asarray(v_jax)
    assert u.shape == v.shape == ones.shape
    drag = sett.rhoSea * (
        sett.waterIceDrag_south if coriolis < 0 else sett.waterIceDrag
    )
    relative_u = u - ocean_velocity[0]
    relative_v = v - ocean_velocity[1]
    speed = np.hypot(relative_u, relative_v)
    np.testing.assert_allclose(
        drag * speed * relative_u - sett.rhoIce * coriolis * v, wind, atol=1e-12
    )
    np.testing.assert_allclose(
        drag * speed * relative_v + sett.rhoIce * coriolis * u,
        0.3 * wind,
        atol=1e-12,
    )
    land = vs._replace(iceMaskU=0 * vs.iceMaskU, iceMaskV=0 * vs.iceMaskV)
    for component in freedrift_solver(land, sett):
        assert component.shape == ones.shape
        np.testing.assert_array_equal(component, 0)
