"""Real setup rollouts retain fields, auxiliary fluxes and multi-step derivatives."""

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import fields, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris import step
from veris._typing import State
from veris.diagnostics import Diagnostics
from veris.setups import artificial, run_dyn, run_growth
from veris.setups.ocean import OceanGeometry, initialize_from_ocean


def make_case(
    case: str, dtype: str
) -> tuple[State, Callable[[State, jax.Array], tuple[State, Diagnostics]]]:
    """Bind real physics, including an independently allocated external ocean grid."""
    options = {"nEVPsteps": 2, "dtype": dtype, "deltatTherm": 600.0}
    if case == "growth":
        initial, conf, phys = run_growth.initialize(settings_overrides=options)
    elif case == "dynamics":
        initial, conf, phys = run_dyn.initialize(6, 8, settings_overrides=options)
    else:
        initial, conf, phys = artificial.initialize(6, 8, settings_overrides=options)
        if case == "ocean":
            geometry = OceanGeometry(
                maskT=initial.iceMask[..., None],
                maskU=initial.iceMaskU[..., None],
                maskV=initial.iceMaskV[..., None],
                ht=initial.R_low,
                coriolis_t=initial.fCori,
                dxt=1 / initial.recip_dxC[:, 0],
                dyt=1 / initial.recip_dyC[0, :],
                dxu=initial.dxU[:, 0],
                dyu=initial.dyU[0, :],
                area_t=1 / initial.recip_rA,
                area_u=1 / initial.recip_rAu,
                area_v=1 / initial.recip_rAv,
            )
            initial, conf, phys = initialize_from_ocean(
                geometry,
                settings_overrides={
                    **options,
                    "use_sharding": False,
                    "deltatDyn": 600.0,
                    "geometrySurfaceTemperature": 260.0,
                },
                state_overrides={
                    f.name: getattr(initial, f.name) for f in fields(initial)
                },
            )
    x, y = jnp.indices(initial.hIceMean.shape, dtype=dtype)
    pattern = 1 + 0.05 * jnp.sin(x + 2 * y)
    initial = replace(initial, hIceMean=initial.hIceMean * pattern)
    if case == "dynamics":
        initial = replace(
            initial,
            Area=0.8 * initial.iceMask,
            hIceMean=initial.hIceMean * initial.iceMask,
        )

    def advance(state: State, forcing: jax.Array) -> tuple[State, Diagnostics]:
        if case == "dynamics":
            state = replace(state, uWind=initial.uWind * forcing)
            return run_dyn.step_with_diagnostics(state, conf, phys)
        if case == "growth":
            state = replace(state, Qnet=jnp.full_like(state.Qnet, 100 * forcing))
            return run_growth.step_with_diagnostics(state, conf, phys)
        return artificial.step_with_diagnostics(state, conf, phys, 100 * forcing)

    return initial, advance


def assert_tree_close(actual: object, expected: object, dtype: str) -> None:
    """Report one compact aggregate error for each failing array leaf."""
    tolerance = 3e-5 if dtype == "float32" else 2e-11
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for index, (a, b) in enumerate(
        zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    ):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all(), (
            f"ERROR nonfinite leaf {index}"
        )
        error = np.abs(a - b)
        assert np.all(error <= tolerance * (1 + np.abs(b))), (
            f"ERROR leaf {index}: max absolute error {error.max():.6g}"
        )


@pytest.mark.parametrize("case", ["artificial", "dynamics", "growth", "ocean"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_setup_rollout_matches_explicit_steps(case: str, dtype: str) -> None:
    """Zero, one and three steps preserve every State field and selected diagnostics."""
    initial, advance = make_case(case, dtype)
    forcing = jnp.asarray([0.8, 1.2, 0.95], dtype=dtype)

    def observe(state: State, aux: Diagnostics) -> dict[str, jax.Array]:
        return {
            "ice": state.hIceMean,
            "freshwater": aux.EmPmR,
            "stress": aux.OceanStressU,
        }

    expected = initial
    history = []
    for index in range(3):
        expected, aux = advance(expected, forcing[index])
        history.append(observe(expected, aux))
        if index == 0:
            one = expected
    for checkpoint in (False, True):
        for count, reference in ((0, initial), (1, one), (3, expected)):
            actual, observed = step(
                initial,
                advance,
                count,
                inputs=forcing[:count],
                checkpoint=checkpoint,
                has_aux=True,
                observe=observe,
            )
            assert_tree_close(actual, reference, dtype)
            assert observed["ice"].shape == (count, *initial.hIceMean.shape)
            if count:
                stacked = jax.tree.map(lambda *v: jnp.stack(v), *history[:count])
                assert_tree_close(observed, stacked, dtype)
            assert_tree_close(
                step(
                    initial,
                    advance,
                    count,
                    inputs=forcing[:count],
                    checkpoint=checkpoint,
                    has_aux=True,
                ),
                reference,
                dtype,
            )


@pytest.mark.parametrize("case", ["artificial", "dynamics", "growth", "ocean"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_real_rollout_state_and_forcing_derivatives(case: str, dtype: str) -> None:
    """Independent spatial directions and time forcing exercise JVP, VJP and FD."""
    initial, advance = make_case(case, dtype)
    x, y = jnp.indices(initial.hIceMean.shape, dtype=dtype)
    direction = (0.3 + 0.07 * jnp.cos(2 * x - y)) * initial.iceMask
    weights = (1 + 0.2 * jnp.sin(x + 3 * y))[2:-2, 2:-2]
    inputs = jnp.asarray([0.85, 1.15], dtype=dtype)
    point = jnp.asarray([0.0, 0.0], dtype=dtype)

    def observe(state: State, aux: Diagnostics) -> jax.Array:
        # Ocean stress makes dynamics forcing sensitivity observable even when
        # transport nearly conserves total ice. Freshwater also exercises aux AD.
        value = state.hIceMean + 10 * state.uIce + aux.EmPmR + aux.OceanStressU
        return jnp.mean(weights * value[2:-2, 2:-2])

    def objective(parameters: jax.Array, checkpoint: bool) -> jax.Array:
        state = replace(initial, hIceMean=initial.hIceMean + parameters[0] * direction)
        forcing = inputs + parameters[1] * jnp.asarray([0.7, -0.2], dtype=dtype)
        _, history = step(
            state,
            advance,
            2,
            inputs=forcing,
            checkpoint=checkpoint,
            has_aux=True,
            observe=observe,
        )
        return history @ jnp.asarray([0.4, 1.0], dtype=dtype)

    results = []
    for checkpoint in (False, True):
        function = jax.jit(partial(objective, checkpoint=checkpoint))
        value, pullback = jax.vjp(function, point)
        gradient = pullback(jnp.ones_like(value))[0]
        assert np.isfinite(gradient).all(), "ERROR nonfinite rollout VJP"
        epsilon = 0.01 if dtype == "float32" else 1e-4
        for axis in range(2):
            tangent = jnp.eye(2, dtype=dtype)[axis]
            _, derivative = jax.jvp(function, (point,), (tangent,))
            finite_difference = (
                function(point + epsilon * tangent)
                - function(point - epsilon * tangent)
            ) / (2 * epsilon)
            assert abs(float(derivative)) > 1e-7, f"ERROR zero sensitivity axis {axis}"
            np.testing.assert_allclose(derivative, gradient[axis], rtol=3e-5, atol=1e-7)
            np.testing.assert_allclose(
                derivative,
                finite_difference,
                rtol=0.02 if dtype == "float32" else 2e-4,
                atol=2e-5 if dtype == "float32" else 2e-8,
                err_msg=f"ERROR {case} {dtype} sensitivity axis {axis}",
            )
        results.append((value, gradient))
    assert_tree_close(results[0], results[1], dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_four_cpu_device_rollout_and_derivatives(dtype: str, tmp_path: Path) -> None:
    """Exercise real halo exchange and scan AD in a fresh four-device CPU process."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        PYTHONPATH=str(root),
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
    )
    logfile = tmp_path / f"rollout-parallel-{dtype}.log"
    with logfile.open("w") as output:
        result = subprocess.run(
            [
                sys.executable,
                str(root / "tests" / "rollout_parallel_probe.py"),
                "--backend",
                "cpu",
                "--dtype",
                dtype,
            ],
            cwd=root,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=360,
            check=False,
        )
    assert result.returncode == 0, (
        f"ERROR four-device {dtype} rollout: {logfile.read_text()[-2500:]}"
    )


def test_growth_rollout_preserves_recursive_heat_fluxes() -> None:
    """The reference growth column reuses computed fluxes on subsequent steps."""
    initial, conf, phys = run_growth.initialize()
    advance = partial(run_growth.step, conf=conf, phys=phys)
    expected = initial
    reset_forcing = initial
    for _ in range(3):
        expected = advance(expected)
        reset_forcing = advance(
            replace(reset_forcing, Qnet=initial.Qnet, Qsw=initial.Qsw)
        )
    actual = step(initial, advance, 3)
    assert_tree_close(actual, expected, "float64")
    assert not np.allclose(actual.Qnet, reset_forcing.Qnet), (
        "ERROR recursive heat-flux test must distinguish prescribed forcing"
    )
