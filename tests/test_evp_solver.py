"""Uniform EVP momentum limits isolate time stepping from spatial stresses."""

import importlib
from dataclasses import replace
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import pytest
from conftest import StateFactory

from veris.configuration import Settings
from veris.physical_constants import PhysicalConstants


class EVPStateFactory(Protocol):
    """Construct dynamically selected fixture fields with optional uniform wind.

    Any is restricted to the varying-field StateFactory result.
    """

    def __call__(self, wind: float = 0) -> Any: ...


@pytest.fixture
def evp_state(state: StateFactory) -> EVPStateFactory:
    """A periodic unit grid with a spatially uniform 900 kg/m² ice column."""

    def build(wind: float = 0) -> Any:
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
def test_unforced_rest_is_exact_evp_fixed_point(
    halo: ModuleType,
    evp_state: EVPStateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    no_slip: bool,
    steps: int,
) -> None:
    solver = importlib.import_module("veris.evp_solver").evp_solver
    vs = evp_state()
    result = solver(vs, replace(sett, noSlip=no_slip, nEVPsteps=steps), phys)
    assert len(result) == 5
    for field in result:
        assert field.shape == (8, 11)
        np.testing.assert_array_equal(field, np.zeros((8, 11)))


@pytest.mark.parametrize("wind", [0.01, 0.1, -0.1])
def test_one_evp_step_uniform_force_matches_mass_drag_balance(
    halo: ModuleType,
    evp_state: EVPStateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    wind: float,
) -> None:
    solver = importlib.import_module("veris.evp_solver").evp_solver
    sett = replace(sett, nEVPsteps=1)
    phys = replace(phys, basalDragK2=0)
    vs = evp_state(wind)
    result = solver(vs, sett, phys)
    denominator = 900 * (sett.evpBeta + 1) / sett.deltatDyn + sett.cDragMin
    for field, expected in zip(
        result, (wind / denominator, -0.5 * wind / denominator, 0, 0, 0)
    ):
        np.testing.assert_allclose(field, expected, atol=1e-14)


def uniform_momentum_subcycles(
    sett: Settings, phys: PhysicalConstants, wind: float, steps: int, beta: float
) -> tuple[float, float, float, float, float]:
    """Solve uniform scalar momentum updates without invoking model kernels.

    Initial velocity and ice strength are zero, so pressure and stress divergence
    vanish. Adaptive relaxation stays at its minimum because viscosity is zero.
    """
    u, v = 0.0, 0.0
    mass_rate = 900 / sett.deltatDyn
    for _ in range(steps):
        drag = max(sett.cDragMin, phys.rhoSea * phys.waterIceDrag * np.hypot(u, v))
        denominator = mass_rate * (beta + 1) + drag
        u = (mass_rate * beta * u + wind) / denominator
        v = (mass_rate * beta * v - 0.5 * wind) / denominator
    return u, v, 0.0, 0.0, 0.0


@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize("wind", [0, 0.01, 0.1, -0.1])
def test_adaptive_evp_uniform_momentum_balance(
    halo: ModuleType,
    evp_state: EVPStateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    steps: int,
    wind: float,
) -> None:
    solver = importlib.import_module("veris.evp_solver").evp_solver
    sett = replace(sett, nEVPsteps=steps, useAdaptiveEVP=True)
    phys = replace(phys, basalDragK2=0)
    result = solver(evp_state(wind), sett, phys)
    expected = uniform_momentum_subcycles(sett, phys, wind, steps, sett.aEVPalphaMin)
    assert len(result) == len(expected)
    for field, value in zip(result, expected):
        assert field.shape == (8, 11)
        np.testing.assert_allclose(field, value, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("wind", [0, 0.1])
def test_evp_residual_diagnostics_preserve_uniform_solution(
    halo: ModuleType,
    evp_state: EVPStateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    adaptive: bool,
    wind: float,
) -> None:
    solver = importlib.import_module("veris.evp_solver").evp_solver
    sett = replace(sett, nEVPsteps=3, useAdaptiveEVP=adaptive)
    phys = replace(phys, basalDragK2=0)
    vs = evp_state(wind)
    # Run the diagnostics path first to reproduce its own failure directly.
    measured = solver(vs, replace(sett, computeEvpResidual=True), phys)
    unmeasured = solver(vs, replace(sett, computeEvpResidual=False), phys)
    beta = sett.aEVPalphaMin if adaptive else sett.evpBeta
    expected = uniform_momentum_subcycles(sett, phys, wind, sett.nEVPsteps, beta)
    assert len(measured) == len(unmeasured) == len(expected)
    for actual, baseline, reference in zip(measured, unmeasured, expected):
        assert actual.shape == baseline.shape == (8, 11)
        np.testing.assert_allclose(actual, baseline, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("principal_stress", [(0, 0, 0), (4, 2, 3)])
@pytest.mark.parametrize("wind", [0.0, 0.1])
def test_printed_evp_residual_matches_interior_velocity_norm(
    halo: ModuleType,
    evp_state: EVPStateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    capsys: pytest.CaptureFixture[str],
    wind: float,
    principal_stress: tuple[int, int, int],
) -> None:
    """Diagnostics must report the actual norm, excluding duplicate halos."""
    import re

    import jax

    module = importlib.import_module("veris.evp_solver")
    sett = replace(sett, nEVPsteps=1, computeEvpResidual=True, printEvpResidual=True)
    phys = replace(phys, basalDragK2=0)
    jax.clear_caches()
    try:
        vs = evp_state(wind)
        s1, s2, s12 = principal_stress
        vs = replace(
            vs,
            sigma1=vs.sigma1 + s1,
            sigma2=vs.sigma2 + s2,
            sigma12=vs.sigma12 + s12,
        )
        result = module.evp_solver(vs, sett, phys)
        jax.block_until_ready(result)
        jax.effects_barrier()
        output = capsys.readouterr().out
        match = re.search(r"evp resU, resSigma: 0 (\S+) (\S+)", output)
        assert match is not None, f"ERROR missing residual diagnostic: {output!r}"
        velocity_norm, stress_norm = map(float, match.groups())
        u, v, *_ = uniform_momentum_subcycles(sett, phys, wind, 1, sett.evpBeta)
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


@pytest.mark.parametrize("partition_axis", [0, 1], ids=["zonal", "meridional"])
@pytest.mark.parametrize("entry_point", ["direct", "dispatcher"])
def test_sharded_evp_residual_sums_all_device_interiors(
    halo: ModuleType,
    evp_state: EVPStateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    capsys: pytest.CaptureFixture[str],
    partition_axis: int,
    entry_point: str,
) -> None:
    """Global diagnostics count each device interior once in either mesh direction."""
    import re

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    module = importlib.import_module("veris.evp_solver")
    count = jax.local_device_count()
    dimensions = (count, 1) if partition_axis == 0 else (1, count)
    mesh = jax.make_mesh(dimensions, ("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))
    wind = 0.1
    sett = replace(sett, nEVPsteps=1, computeEvpResidual=True, printEvpResidual=True)
    phys = replace(phys, basalDragK2=0)
    vs = evp_state(wind)
    vs = replace(vs, sigma1=vs.sigma1 + 4, sigma2=vs.sigma2 + 2, sigma12=vs.sigma12 + 3)
    # Each shard owns a full 8x11 local array, including its two-cell halos.
    # Uniform fields make serial and exchanged halos identical in this oracle.
    distributed = jax.tree.map(
        lambda field: jax.device_put(jnp.tile(field, dimensions), sharding), vs
    )
    jax.clear_caches()
    try:
        solver = (
            module.evp_solver
            if entry_point == "direct"
            else importlib.import_module("veris.dynsolver").IceVelocities
        )
        solve = jax.shard_map(
            lambda local: solver(local, sett, phys, axis_names=("x", "y")),
            mesh=mesh,
            in_specs=P("x", "y"),
            out_specs=P("x", "y"),
        )
        result = solve(distributed)
        jax.block_until_ready(result)
        jax.effects_barrier()
        output = capsys.readouterr().out
        matches = re.findall(r"evp resU, resSigma: 0 (\S+) (\S+)", output)
        assert matches, f"ERROR missing distributed residual diagnostic: {output!r}"
        u, v, *_ = uniform_momentum_subcycles(sett, phys, wind, 1, sett.evpBeta)
        expected_velocity = count * 4 * 7 * sett.evpBeta**2 * (u * u + v * v)
        # Principal stresses (4,2) give physical stresses (3,1); shear is 3.
        expected_stress = count * 4 * 7 * (3**2 + 1**2 + 3**2)
        for velocity_norm, stress_norm in matches:
            np.testing.assert_allclose(
                [float(velocity_norm), float(stress_norm)],
                [expected_velocity, expected_stress],
                rtol=1e-6,
                atol=1e-14,
            )
        np.testing.assert_allclose(result[0], u, atol=1e-14)
        np.testing.assert_allclose(result[1], v, atol=1e-14)
    finally:
        jax.clear_caches()


@pytest.mark.parametrize("relaxation", [0.5, 2.0])
def test_evp_configured_normal_relaxation_damps_uniform_stress(
    halo: ModuleType,
    evp_state: EVPStateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    relaxation: float,
) -> None:
    """Independent normal stress damping is read from initialized settings."""
    solver = importlib.import_module("veris.evp_solver").evp_solver
    sett = replace(sett, nEVPsteps=1, evpStressRelaxation=relaxation)
    vs = evp_state()
    vs = replace(vs, sigma1=vs.sigma1 + 4, sigma2=vs.sigma2 + 2, sigma12=vs.sigma12 + 3)
    result = solver(vs, sett, phys)
    damping = (sett.evpAlpha - relaxation) / sett.evpAlpha
    for field, initial in zip(result[2:], (4, 2, 3)):
        np.testing.assert_allclose(field, initial * damping, rtol=1e-14, atol=1e-15)
