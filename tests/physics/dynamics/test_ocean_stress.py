"""Ice-ocean stresses must oppose relative motion and rotate by turning angle."""

import importlib
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest
from conftest import StateFactory

from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants


@pytest.mark.parametrize("hemisphere", [-1, 1])
@pytest.mark.parametrize("angle", [0, 25])
@pytest.mark.parametrize("relative", [(0, 0), (0.2, -0.1)])
def test_uniform_ocean_stress_rotation(
    halo: ModuleType,
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    hemisphere: int,
    angle: int,
    relative: tuple[float, float],
) -> None:
    ocean = importlib.import_module("veris.physics.dynamics.ocean_stress")
    phys = replace(phys, waterTurnAngle=angle, waterIceDrag_south=0.007)
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
        phys.cDragMin,
        phys.rhoSea
        * np.hypot(du, dv)
        * (phys.waterIceDrag_south if hemisphere < 0 else phys.waterIceDrag),
    )
    radians = np.deg2rad(angle)
    expected = (
        drag * (np.cos(radians) * du - hemisphere * np.sin(radians) * dv),
        drag * (np.cos(radians) * dv + hemisphere * np.sin(radians) * du),
    )
    for actual, scalar in zip(ocean.OceanStressUV(vs, conf, phys), expected):
        assert actual.shape == ones.shape
        np.testing.assert_allclose(actual, scalar, atol=1e-14)


@pytest.mark.parametrize("hemisphere", [-1, 1])
def test_nonuniform_turning_stress_uses_orthogonal_face_average(
    state: StateFactory, conf: Configuration, phys: PhysicalConstants, hemisphere: int
) -> None:
    """A linear cross velocity is sampled at each west/south face midpoint."""
    from veris.physics.dynamics.ocean_stress import OceanStressUV

    x, y = np.indices((12, 12))
    one, zero = np.ones((12, 12)), np.zeros((12, 12))
    config = replace(conf, nx=8, ny=8, noSlip=False)
    parameters = phys.with_overrides(waterTurnAngle=30.0, cDragMin=2.0)
    vs = state(
        uIce=0.02 * y,
        vIce=0.01 * x,
        uOcean=zero,
        vOcean=zero,
        maskInU=one,
        maskInV=one,
        iceMask=one,
        fCori=hemisphere * one,
    )
    u, v = OceanStressUV(vs, config, parameters)
    cosine, sine = np.cos(np.pi / 6), np.sin(np.pi / 6)
    expected = (
        2 * cosine * 0.02 * 5 - hemisphere * 2 * sine * 0.01 * 4.5,
        2 * cosine * 0.01 * 5 + hemisphere * 2 * sine * 0.02 * 4.5,
    )
    np.testing.assert_allclose([u[5, 5], v[5, 5]], expected, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("hemisphere", [-1, 1])
def test_evp_turning_force_matches_independent_uniform_balance(hemisphere: int) -> None:
    """One EVP iteration uses the opposite ocean cross traction; angle AD agrees."""
    import jax
    import jax.numpy as jnp

    import veris
    from veris.numerics.evp import evp_solver

    definition = veris.Experiment.builtin(
        "growth",
        grid=veris.Grid(8, 8),
        settings={
            "nEVPsteps": 1,
            "deltatDyn": 300.0,
            "deltatTherm": 300.0,
            "noSlip": False,
            "useAdaptiveEVP": False,
        },
    )
    vs, config, parameters = veris.initialize_experiment(definition)
    one, zero = jnp.ones_like(vs.Area), jnp.zeros_like(vs.Area)
    vs = replace(
        vs,
        uIce=zero,
        vIce=one * 0.1,
        uOcean=zero,
        vOcean=zero,
        SeaIceMassC=900 * one,
        SeaIceMassU=900 * one,
        SeaIceMassV=900 * one,
        Area=one,
        AreaW=one,
        AreaS=one,
        fCori=hemisphere * one * 1e-20,
        SeaIceStrength=zero,
        WindForcingX=zero,
        WindForcingY=zero,
        sigma1=zero,
        sigma2=zero,
        sigma12=zero,
        iceMask=one,
        iceMaskU=one,
        iceMaskV=one,
        maskInU=one,
        maskInV=one,
    )
    parameters = parameters.with_overrides(cDragMin=2.0, sideDragCoeff=0.0)

    def predict(angle: jax.Array) -> jax.Array:
        return evp_solver(vs, config, parameters.with_overrides(waterTurnAngle=angle))[
            0
        ][5, 5]

    angle = jnp.asarray(30.0)
    denominator = 900 * (config.evpBeta + 1) / 300 + 2 * np.cos(np.pi / 6)
    expected = hemisphere * 2 * np.sin(np.pi / 6) * 0.1 / denominator
    np.testing.assert_allclose(predict(angle), expected, rtol=1e-12, atol=1e-16)
    derivative = jax.grad(predict)(angle)
    _, tangent = jax.jvp(predict, (angle,), (jnp.ones_like(angle),))
    finite = (predict(angle + 1e-3) - predict(angle - 1e-3)) / (2e-3)
    np.testing.assert_allclose([derivative, tangent], finite, rtol=1e-8, atol=1e-12)


def test_freedrift_rejects_unsupported_turning_angle_before_host_allocation() -> None:
    import veris

    with pytest.raises(ValueError, match="free.drift.*turning angle"):
        veris.Experiment.builtin(
            "island",
            settings={"useFreedrift": True, "useEVP": False},
            constants={"waterTurnAngle": 30.0},
        )


def test_traced_freedrift_turning_angle_has_explicit_failure_status() -> None:
    import jax
    import jax.numpy as jnp

    import veris

    definition = veris.Experiment.builtin(
        "growth",
        grid=veris.Grid(2, 2),
        settings={"useFreedrift": True, "useEVP": False},
    )
    state, config, parameters = veris.initialize_experiment(definition)
    step = jax.jit(
        lambda angle: veris.advance(
            state,
            config,
            parameters.with_overrides(waterTurnAngle=angle),
            physics="dynamics",
        )
    )
    _, valid = step(jnp.asarray(0.0))
    _, rejected = step(jnp.asarray(30.0))
    np.testing.assert_array_equal(valid.physics_status, 0)
    np.testing.assert_array_equal(rejected.physics_status, 1)
