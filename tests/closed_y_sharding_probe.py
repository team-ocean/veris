"""Fresh-process closed-y communication, coupled stepping and AD checks.

Run with four virtual CPUs using XLA_FLAGS=--xla_force_host_platform_device_count=4
and ``python tests/closed_y_sharding_probe.py``. ``--backend gpu`` reuses the
same checks on the available GPU devices, including two-device y partitions.
The halo oracle uses global clipped coordinates and explicit wall locations.
"""

from dataclasses import fields, replace

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

from veris import (
    Execution,
    Experiment,
    Forcing,
    Grid,
    advance,
    initialize_experiment,
    step,
)
from veris.grid.halos import fill_overlap, fill_overlap_uv, remove_halos
from veris.model.configuration import Configuration
from veris.model.types import State


def host_array(value: ArrayLike) -> np.ndarray:
    """Gather non-addressable test fields on every distributed controller."""
    if (
        isinstance(value, jax.Array)
        and not value.is_fully_addressable
        and not value.is_fully_replicated
    ):
        from jax.experimental import multihost_utils

        value = multihost_utils.process_allgather(value, tiled=True)
    return np.asarray(value)


def placed(value: ArrayLike, sharding: NamedSharding) -> jax.Array:
    """Keep local placement unchanged; construct addressable distributed inputs."""
    if jax.process_count() == 1:
        return jax.device_put(value, sharding)
    source = np.asarray(value)
    return jax.make_array_from_callback(
        source.shape, sharding, lambda index: source[index]
    )


def compare(
    actual: ArrayLike, expected: ArrayLike, label: str, tolerance: float = 2e-10
) -> None:
    """Report finite aggregate discrepancies without printing large arrays."""
    left, right = host_array(actual), host_array(expected)
    error = np.abs(left - right)
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise AssertionError(f"ERROR {label}: nonfinite values")
    if not np.all(error <= tolerance * (1 + np.abs(right))):
        raise AssertionError(f"ERROR {label}: max absolute error={error.max():.6g}")


def check_halos(mesh: Mesh, ny: int = 5) -> None:
    """Check every cell and exact forward/reverse linear maps for all wall policies."""
    px, py = mesh.shape["x"], mesh.shape["y"]
    nx = 3
    gx, gy = px * nx, py * ny
    shape = (px * (nx + 4), py * (ny + 4))
    data = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1
    direction = np.sin(data)
    weights = np.cos(data / 7)
    conf = Configuration(nx=nx, ny=ny, use_sharding=True, enable_cyclic_y=False)
    sharding = NamedSharding(mesh, P("x", "y"))
    for mode in ("edge", "zero", "normal", "shear"):
        expected = np.zeros(shape)
        tangent_expected = np.zeros(shape)
        gradient_expected = np.zeros(shape)
        for rx in range(px):
            for ry in range(py):
                for i, j in np.ndindex((nx + 4, ny + 4)):
                    global_x, global_y = (rx * nx + i - 2) % gx, ry * ny + j - 2
                    out = (rx * (nx + 4) + i, ry * (ny + 4) + j)
                    closed = (global_y < 0 or global_y >= gy) and mode != "edge"
                    if mode == "shear":
                        closed = global_y < 0 or global_y > gy
                    closed |= global_y == 0 and mode == "normal"
                    if closed:
                        continue
                    if mode == "shear" and global_y == gy:
                        source_y = shape[1] - 2
                    else:
                        global_y = np.clip(global_y, 0, gy - 1)
                        source_y = (global_y // ny) * (ny + 4) + 2 + global_y % ny
                    source = (
                        (global_x // nx) * (nx + 4) + 2 + global_x % nx,
                        source_y,
                    )
                    expected[out] = data[source]
                    tangent_expected[out] = direction[source]
                    gradient_expected[source] += weights[out]

        def refresh(value: jax.Array, mode: str = mode) -> jax.Array:
            if mode == "edge":
                return fill_overlap(value, conf)
            if mode == "shear":
                return fill_overlap(value, conf, boundary="shear")
            return fill_overlap_uv(value, value, conf)[mode == "normal"]

        with jax.set_mesh(mesh):
            value = placed(data, sharding)
            tangent = placed(direction, sharding)
            actual, derivative = jax.jvp(refresh, (value,), (tangent,))
            _, pullback = jax.vjp(refresh, value)
            gradient = pullback(placed(weights, sharding))[0]
            compare(actual, expected, f"{px}x{py} {mode} halo", 0)
            compare(refresh(actual), expected, f"{px}x{py} {mode} idempotence", 0)
            compare(derivative, tangent_expected, f"{px}x{py} {mode} JVP", 0)
            compare(gradient, gradient_expected, f"{px}x{py} {mode} VJP")
    print(
        f"closed-y halos: {px}x{py} ny={ny} edge/zero/normal/shear values, JVP and VJP pass",
        flush=True,
    )


def check_coupled(mesh: Mesh, no_slip: bool, dtype: str) -> None:
    """Compare two coupled steps and spatially weighted initial/forcing AD."""
    px, py = mesh.shape["x"], mesh.shape["y"]
    nx, ny = 4 * px, 5 * py
    tolerance = 8e-5 if dtype == "float32" else 3e-10
    options = {
        "dtype": dtype,
        "enable_cyclic_y": False,
        "noSlip": no_slip,
        "nEVPsteps": 2,
        "deltatDyn": 600.0,
        "deltatTherm": 600.0,
    }
    x, y = np.indices((nx, ny), dtype=dtype)
    # Original storage coordinates started at two on each physical axis.
    x, y = x + 2, y + 2
    definition = Experiment.builtin(
        "island",
        grid=Grid(nx, ny),
        settings=options,
        initial_conditions={
            "hIceMean": 0.8 + 0.03 * np.sin(x + y),
            "Area": 0.7,
        },
        forcing=Forcing(
            constant={
                "uWind": 4 + 0.3 * np.cos(y),
                "vWind": 2 + 0.2 * np.sin(x + y),
                "wSpeed": 5.0,
            }
        ),
    )
    serial, conf, phys = initialize_experiment(definition)
    assert np.all(np.asarray(serial.iceMask)[2:-2, 2] == 1)
    assert np.all(np.asarray(serial.iceMask)[2:-2, -3] == 1)
    with jax.set_mesh(mesh):
        parallel, local_conf, local_phys = initialize_experiment(
            replace(
                definition,
                execution=Execution(
                    backend=mesh.devices.flat[0].platform, partitions=(px, py)
                ),
            ),
            mesh=mesh,
        )
    point = jnp.zeros(3, dtype=dtype)
    spatial_weights = 1 + 0.2 * jnp.sin(
        jnp.arange(nx)[:, None] + 2 * jnp.arange(ny)[None, :]
    )

    def evolve(
        parameters: jax.Array, sharded: bool, initial: State | None = None
    ) -> State:
        state, settings, physical = (
            (parallel, local_conf, local_phys) if sharded else (serial, conf, phys)
        )
        if initial is not None:
            state = initial
        state = replace(state, hIceMean=state.hIceMean * (1 + parameters[0]))
        physical = physical.with_overrides(
            airIceDrag=physical.airIceDrag * (1 + parameters[2])
        )

        def transition(current: State, cooling: jax.Array) -> State:
            return advance(current, settings, physical, {"Qnet": cooling, "Qsw": 0.0})[
                0
            ]

        return step(
            state,
            transition,
            2,
            inputs=jnp.asarray([80.0, 110.0], dtype=dtype) + 50 * parameters[1],
        )

    def objective(
        parameters: jax.Array, sharded: bool, initial: State | None = None
    ) -> jax.Array:
        result = evolve(parameters, sharded, initial)
        value = result.hIceMean + 10 * result.vIce
        physical = remove_halos(value, mesh) if sharded else value[2:-2, 2:-2]
        return jnp.mean(spatial_weights * physical)

    reference = evolve(point, False)
    # Keep the independent oracle process-local. Below, point becomes a global
    # replicated array; using it for the serial oracle can enqueue collectives
    # while the distributed pullback is still running.
    expected_value = np.asarray(objective(point, False))
    expected_gradient = np.asarray(jax.grad(lambda p: objective(p, False))(point))
    with jax.set_mesh(mesh):
        replicated = NamedSharding(mesh, P())
        point = placed(point, replicated)
        actual = evolve(point, True)
        for field in fields(actual):
            compare(
                remove_halos(getattr(actual, field.name), mesh),
                getattr(reference, field.name)[2:-2, 2:-2],
                f"coupled noSlip={no_slip} {field.name}",
                tolerance,
            )
        # The north corner shear is a physical boundary unknown stored in a
        # halo column, so interior-only comparisons would omit its dynamics.
        local_nx = nx // px
        stored_shear = host_array(actual.sigma12)
        north_shear = np.concatenate(
            [
                stored_shear[
                    rx * (local_nx + 4) + 2 : rx * (local_nx + 4) + local_nx + 2,
                    -2,
                ]
                for rx in range(px)
            ]
        )
        compare(north_shear, reference.sigma12[2:-2, -2], "north wall shear", tolerance)
        stored_velocity = host_array(actual.vIce)
        compare(stored_velocity[:, :3], 0, "south wall velocity", 0)
        compare(stored_velocity[:, -2:], 0, "north wall velocity", 0)
        distributed = jax.process_count() > 1
        if distributed:

            def explicit_objective(p: jax.Array, initial: State) -> jax.Array:
                frozen = jax.tree.map(jax.lax.stop_gradient, initial)
                return objective(p, True, frozen)

            function = jax.jit(explicit_objective)
            arguments = (point, parallel)
            state_tangent = jax.tree.map(jnp.zeros_like, parallel)
        else:
            function = jax.jit(lambda p: objective(p, True))
            arguments = (point,)
            state_tangent = None
        value, pullback = jax.vjp(function, *arguments)
        gradient = pullback(jnp.ones_like(value))[0]
        compare(value, expected_value, "coupled scalar objective", tolerance)
        compare(gradient, expected_gradient, "coupled VJP serial/sharded", tolerance)
        if distributed:
            # Exercise another cold executable without adding synchronization
            # between the primal and pullback. Replayed nested communication
            # previously collided with independent adjoint halo exchanges.
            jax.clear_caches()
            repeated_value, repeated_pullback = jax.vjp(function, *arguments)
            repeated_gradient = repeated_pullback(jnp.ones_like(repeated_value))[0]
            compare(
                repeated_value, expected_value, "fresh coupled objective", tolerance
            )
            compare(
                repeated_gradient, expected_gradient, "fresh coupled VJP", tolerance
            )
        for axis in range(3):
            direction = placed(jnp.eye(3, dtype=dtype)[axis], replicated)
            tangents = (direction, state_tangent) if distributed else (direction,)
            _, tangent = jax.jvp(function, arguments, tangents)
            compare(tangent, gradient[axis], f"coupled JVP/VJP axis={axis}", tolerance)
            assert abs(float(tangent)) > 1e-8, "ERROR zero coupled sensitivity"
            delta = 0.01 if dtype == "float32" else 1e-4
            finite_difference = (
                function(point + delta * direction, *arguments[1:])
                - function(point - delta * direction, *arguments[1:])
            ) / (2 * delta)
            fd_tolerance = 2e-4 if dtype == "float32" else 2e-7
            compare(
                tangent,
                finite_difference,
                f"coupled finite difference axis={axis}",
                fd_tolerance,
            )
    print(
        f"closed-y coupled: noSlip={no_slip}, two steps, all fields and AD pass",
        flush=True,
    )


@click.command(help=__doc__)
@click.option("--backend", type=click.Choice(["cpu", "gpu"]), default="cpu")
@click.option("--dtype", type=click.Choice(["float32", "float64"]), default="float64")
@click.option("--halos-only", is_flag=True, help="Run only communication and its AD.")
@click.option(
    "--distributed", is_flag=True, help="Initialize the launched controllers."
)
def main(backend: str, dtype: str, halos_only: bool, distributed: bool) -> None:
    """Exercise all device-grid orientations and both wall-slip choices."""
    from veris.execution.runtime import distributed_options

    jax.config.update("jax_enable_x64", True)
    if distributed:
        import os

        jax.config.update("jax_enable_compilation_cache", False)
        if backend == "cpu":
            jax.config.update("jax_num_cpu_devices", 1)
            jax.config.update("jax_cpu_collectives_implementation", "gloo")
        options = distributed_options(os.environ, backend)
        assert options is not None, "ERROR distributed probe needs launch controls"
        jax.distributed.initialize(**options)
    try:
        run_checks(backend, dtype, halos_only)
    finally:
        if distributed:
            jax.distributed.shutdown()


def run_checks(backend: str, dtype: str, halos_only: bool) -> None:
    """Run the same halo prelude and coupled checks on local or global meshes."""
    from veris.execution.runtime import create_mesh

    devices = jax.devices(backend)
    count = len(devices)
    assert count >= 2, "ERROR at least two devices required"
    layouts = [(1, count), (count, 1)]
    if count >= 4 and count % 2 == 0:
        layouts.append((2, count // 2))

    def mesh_for(px: int, py: int) -> Mesh:
        if jax.process_count() > 1:
            return create_mesh((px, py), backend)
        return Mesh(np.asarray(devices, dtype=object).reshape(px, py), ("x", "y"))

    with jax.default_device(jax.local_devices(backend=backend)[0]):
        for px, py in layouts:
            mesh = mesh_for(px, py)
            for local_ny in (2, 5):
                check_halos(mesh, local_ny)
        if not halos_only:
            px, py = layouts[-1] if count >= 4 else (1, count)
            mesh = mesh_for(px, py)
            for no_slip in (False, True):
                check_coupled(mesh, no_slip, dtype)


if __name__ == "__main__":
    main()
