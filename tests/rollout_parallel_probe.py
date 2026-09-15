"""Real sharded scan/AD equivalence on CPU or GPU; run outside pytest.

Example: XLA_FLAGS=--xla_force_host_platform_device_count=4 python
 tests/rollout_parallel_probe.py --backend cpu --dtype float64
Use two/four devices to exercise exchanged partition boundaries. A single GPU
also exercises the explicit mesh path, but not communication between devices.
"""

from dataclasses import fields, replace
from functools import partial

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from veris import step
from veris._typing import State
from veris.setups import artificial, run_dyn, run_parallel


def check_parallel_rollout(backend: str, dtype: str) -> None:
    """Compare all physical fields and initial/forcing derivatives to serial scans."""
    for case in ("dynamics", "coupled"):
        _check_case(case, backend, dtype)


def _check_case(case: str, backend: str, dtype: str) -> None:
    """Bind one experiment before tracing checkpointed and ordinary objectives."""
    devices = jax.devices(backend)
    px = 2 if len(devices) % 2 == 0 else 1
    py = len(devices) // px
    mesh = run_parallel.create_mesh((px, py), backend)
    nx, ny = 6 * px, 8 * py
    options = {"dtype": dtype, "nEVPsteps": 2, "deltatTherm": 600.0}
    tolerance = 5e-5 if dtype == "float32" else 2e-10

    def compare(actual: jax.Array, expected: jax.Array, label: str) -> None:
        left, right = np.asarray(actual), np.asarray(expected)
        error = np.abs(left - right)
        assert np.isfinite(left).all() and np.isfinite(right).all(), (
            f"ERROR nonfinite {label}"
        )
        assert np.all(error <= tolerance * (1 + np.abs(right))), (
            f"ERROR {label}: max absolute difference {error.max():.6g}"
        )

    serial, conf, phys = run_dyn.initialize(nx, ny, settings_overrides=options)
    with jax.set_mesh(mesh):
        parallel, local_conf, local_phys = run_dyn.initialize(
            nx, ny, mesh=mesh, settings_overrides=options
        )

    def prepare(state: State) -> State:
        # Interior concentration stays below ridging's nonsmooth threshold;
        # prescribed cold atmosphere gives the coupled setup valid growth.
        return replace(
            state,
            Area=0.8 * state.iceMask,
            hIceMean=(0.8 + 0.02 * jnp.sin(state.uWind)) * state.iceMask,
            hSnowMean=0.05 * state.iceMask,
            TSurf=jnp.full_like(state.TSurf, 260.0),
            theta=jnp.full_like(state.theta, phys.celsius2K + phys.tempFrz),
            ocSalt=jnp.full_like(state.ocSalt, 34.7),
            ATemp=jnp.full_like(state.ATemp, 260.0),
            LWdown=jnp.full_like(state.LWdown, phys.stefBoltz * 260.0**4),
            wSpeed=jnp.sqrt(state.uWind**2 + state.vWind**2),
        )

    serial = prepare(serial)
    with jax.set_mesh(mesh):
        parallel = prepare(parallel)
    forcing = jnp.asarray([80.0, 115.0], dtype=dtype)
    point = jnp.zeros(2, dtype=dtype)

    def evolve(parameters: jax.Array, sharded: bool, checkpoint: bool) -> State:
        initial = parallel if sharded else serial
        settings = local_conf if sharded else conf
        physical = local_phys if sharded else phys
        state = replace(initial, hIceMean=initial.hIceMean * (1 + parameters[0]))
        fluxes = forcing + parameters[1] * jnp.asarray([70.0, -20.0], dtype=dtype)

        def advance(current: State, cooling: jax.Array) -> State:
            if case == "coupled":
                # artificial.step shard_map explicitly broadcasts this scalar
                # with P(), while State fields retain P('x', 'y').
                return artificial.step(current, settings, physical, cooling)
            current = replace(current, uWind=initial.uWind * cooling / 100)
            return run_dyn.step(current, settings, physical)

        return step(state, advance, 2, inputs=fluxes, checkpoint=checkpoint)

    def objective(parameters: jax.Array, sharded: bool, checkpoint: bool) -> jax.Array:
        final = evolve(parameters, sharded, checkpoint)
        value = final.hIceMean + 10 * final.uIce
        value = run_parallel.remove_halos(value, mesh) if sharded else value[2:-2, 2:-2]
        x, y = jnp.indices((nx, ny), dtype=dtype)
        weights = 1 + 0.2 * jnp.sin(x + 2 * y)
        return jnp.mean(weights * value)

    reference = evolve(point, False, False)
    reference_gradient = jax.grad(partial(objective, sharded=False, checkpoint=False))(
        point
    )
    for checkpoint in (False, True):
        with jax.set_mesh(mesh):
            replicated_point = jax.device_put(point, NamedSharding(mesh, P()))
            actual = evolve(replicated_point, True, checkpoint)
            for metadata in fields(State):
                compare(
                    run_parallel.remove_halos(getattr(actual, metadata.name), mesh),
                    getattr(reference, metadata.name)[2:-2, 2:-2],
                    f"{case} checkpoint={checkpoint} {metadata.name}",
                )
            function = jax.jit(partial(objective, sharded=True, checkpoint=checkpoint))
            value, pullback = jax.vjp(function, replicated_point)
            gradient = pullback(jnp.ones_like(value))[0]
            compare(gradient, reference_gradient, f"{case} serial/sharded VJP")
            for axis in range(2):
                tangent = jax.device_put(
                    jnp.eye(2, dtype=dtype)[axis], NamedSharding(mesh, P())
                )
                _, derivative = jax.jvp(function, (replicated_point,), (tangent,))
                compare(derivative, gradient[axis], f"{case} JVP/VJP axis {axis}")
                assert abs(float(derivative)) > 1e-7, "ERROR zero sharded sensitivity"
                epsilon = 0.01 if dtype == "float32" else 1e-4
                fd = (
                    function(replicated_point + epsilon * tangent)
                    - function(replicated_point - epsilon * tangent)
                ) / (2 * epsilon)
                np.testing.assert_allclose(
                    derivative,
                    fd,
                    rtol=0.02 if dtype == "float32" else 2e-4,
                    atol=2e-5 if dtype == "float32" else 2e-8,
                    err_msg=f"ERROR {case} sharded FD axis {axis}",
                )
    print(f"{case}: {len(devices)} {backend} devices, {dtype}, scan fields and AD pass")


@click.command(help=__doc__)
@click.option("--backend", type=click.Choice(["cpu", "gpu"]), default="cpu")
@click.option("--dtype", type=click.Choice(["float32", "float64"]), default="float64")
def main(backend: str, dtype: str) -> None:
    """Run distributed forward and derivative checks on the selected backend."""
    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices(backend)[0]):
        check_parallel_rollout(backend, dtype)


if __name__ == "__main__":
    main()
