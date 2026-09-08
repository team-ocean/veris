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


def uniform_momentum_subcycles(sett, wind, steps, beta):
    """Solve uniform scalar momentum updates without invoking model kernels.

    Initial velocity and ice strength are zero, so pressure and stress divergence
    vanish. Adaptive relaxation stays at its minimum because viscosity is zero.
    """
    u, v = 0.0, 0.0
    mass_rate = 900 / sett.deltatDyn
    for _ in range(steps):
        drag = max(sett.cDragMin, sett.rhoSea * sett.waterIceDrag * np.hypot(u, v))
        denominator = mass_rate * (beta + 1) + drag
        u = (mass_rate * beta * u + wind) / denominator
        v = (mass_rate * beta * v - 0.5 * wind) / denominator
    return u, v, 0.0, 0.0, 0.0


@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize("wind", [0, 0.01, 0.1, -0.1])
def test_adaptive_evp_uniform_momentum_balance(halo, evp_state, sett, steps, wind):
    solver = importlib.import_module("veris.evp_solver").evp_solver
    sett = sett._replace(
        nEVPsteps=steps,
        useAdaptiveEVP=True,
        basalDragK2=0,
        cosWat=1.0,
        sinWat=0.0,
    )
    result = solver(evp_state(wind), sett)
    expected = uniform_momentum_subcycles(sett, wind, steps, sett.aEVPalphaMin)
    assert len(result) == len(expected)
    for field, value in zip(result, expected):
        assert field.shape == (8, 11)
        np.testing.assert_allclose(field, value, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("wind", [0, 0.1])
def test_evp_residual_diagnostics_preserve_uniform_solution(
    halo, evp_state, sett, adaptive, wind
):
    solver = importlib.import_module("veris.evp_solver").evp_solver
    sett = sett._replace(
        nEVPsteps=3,
        useAdaptiveEVP=adaptive,
        basalDragK2=0,
        cosWat=1.0,
        sinWat=0.0,
    )
    vs = evp_state(wind)
    # Run the diagnostics path first to reproduce its own failure directly.
    measured = solver(vs, sett._replace(computeEvpResidual=True))
    unmeasured = solver(vs, sett._replace(computeEvpResidual=False))
    beta = sett.aEVPalphaMin if adaptive else sett.evpBeta
    expected = uniform_momentum_subcycles(sett, wind, sett.nEVPsteps, beta)
    assert len(measured) == len(unmeasured) == len(expected)
    for actual, baseline, reference in zip(measured, unmeasured, expected):
        assert actual.shape == baseline.shape == (8, 11)
        np.testing.assert_allclose(actual, baseline, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("principal_stress", [(0, 0, 0), (4, 2, 3)])
@pytest.mark.parametrize("wind", [0.0, 0.1])
def test_printed_evp_residual_matches_interior_velocity_norm(
    halo, evp_state, sett, monkeypatch, capsys, wind, principal_stress
):
    """Diagnostics must report the actual norm, excluding duplicate halos."""
    import re

    import jax

    module = importlib.import_module("veris.evp_solver")
    sett = sett._replace(nEVPsteps=1, computeEvpResidual=True, basalDragK2=0)
    jax.clear_caches()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module, "printEvpResidual", True)
            vs = evp_state(wind)
            s1, s2, s12 = principal_stress
            vs = vs._replace(
                sigma1=vs.sigma1 + s1, sigma2=vs.sigma2 + s2, sigma12=vs.sigma12 + s12
            )
            result = module.evp_solver(vs, sett)
            jax.block_until_ready(result)
            jax.effects_barrier()
        output = capsys.readouterr().out
        match = re.search(r"evp resU, resSigma: 0 (\S+) (\S+)", output)
        assert match is not None, f"ERROR missing residual diagnostic: {output!r}"
        velocity_norm, stress_norm = map(float, match.groups())
        u, v, *_ = uniform_momentum_subcycles(sett, wind, 1, sett.evpBeta)
        expected = 4 * 7 * sett.evpBeta**2 * (u * u + v * v)
        np.testing.assert_allclose(velocity_norm, expected, rtol=1e-6, atol=1e-14)
        # With zero strain/strength, one relaxation step changes each physical
        # stress by -sigma/alpha; scaling by alpha recovers its original norm.
        expected_stress_norm = (
            4 * 7 * ((0.5 * (s1 + s2)) ** 2 + (0.5 * (s1 - s2)) ** 2 + s12**2)
        )
        np.testing.assert_allclose(
            stress_norm, expected_stress_norm, rtol=1e-6, atol=1e-14
        )
    finally:
        jax.clear_caches()
