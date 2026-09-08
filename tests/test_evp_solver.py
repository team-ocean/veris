"""Uniform EVP momentum limits isolate time stepping from spatial stresses."""

import importlib

import numpy as np
import pytest


@pytest.fixture
def evp_state(state):
    """A periodic unit grid with a spatially uniform 900 kg/m² ice column."""

    def build(wind=0):
        ones = np.ones((8, 11))
        fields = {
            name: ones
            for name in (
                "iceMask",
                "iceMaskU",
                "iceMaskV",
                "maskInC",
                "maskInU",
                "maskInV",
                "recip_dxU",
                "recip_dyV",
                "recip_dyU",
                "recip_dxV",
                "rAz",
                "recip_rA",
                "dyV",
                "dxV",
                "dxU",
                "dyU",
                "recip_rAu",
                "recip_rAv",
                "Area",
                "AreaW",
                "AreaS",
                "hIceMean",
            )
        }
        fields.update(
            {
                name: 0 * ones
                for name in (
                    "uIce",
                    "vIce",
                    "uOcean",
                    "vOcean",
                    "sigma1",
                    "sigma2",
                    "sigma12",
                    "k1AtC",
                    "k2AtC",
                    "k1AtZ",
                    "k2AtZ",
                    "SeaIceStrength",
                    "fCori",
                )
            }
        )
        fields.update(
            {name: 900 * ones for name in ("SeaIceMassC", "SeaIceMassU", "SeaIceMassV")}
        )
        fields.update(
            R_low=-1000 * ones,
            WindForcingX=wind * ones,
            WindForcingY=-0.5 * wind * ones,
        )
        return state(**fields)

    return build


@pytest.mark.parametrize("no_slip", [False, True])
@pytest.mark.parametrize("steps", [1, 4])
def test_unforced_rest_is_exact_evp_fixed_point(halo, evp_state, sett, no_slip, steps):
    solver = importlib.import_module("veris.evp_solver").evp_solver
    vs = evp_state()
    result = solver(vs, sett._replace(noSlip=no_slip, nEVPsteps=steps))
    assert len(result) == 5
    for field in result:
        assert field.shape == (8, 11)
        np.testing.assert_array_equal(field, np.zeros((8, 11)))


@pytest.mark.parametrize("wind", [0.01, 0.1, -0.1])
def test_one_evp_step_uniform_force_matches_mass_drag_balance(
    halo, evp_state, sett, wind
):
    solver = importlib.import_module("veris.evp_solver").evp_solver
    sett = sett._replace(nEVPsteps=1, basalDragK2=0, cosWat=1.0, sinWat=0.0)
    vs = evp_state(wind)
    result = solver(vs, sett)
    denominator = 900 * (sett.evpBeta + 1) / sett.deltatDyn + sett.cDragMin
    for field, expected in zip(
        result, (wind / denominator, -0.5 * wind / denominator, 0, 0, 0)
    ):
        np.testing.assert_allclose(field, expected, atol=1e-14)
