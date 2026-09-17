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

from veris._typing import State
from veris.configuration import Configuration
from veris.fill_overlap import fill_overlap, fill_overlap_uv, fill_state_overlap
from veris.initialization import initialize
from veris.setups.island import initialize as initialize_island
from veris.setups.island import step
from veris.setups.run_parallel import remove_halos


def compare(
    actual: ArrayLike, expected: ArrayLike, label: str, tolerance: float = 2e-10
) -> None:
    """Report finite aggregate discrepancies without printing large arrays."""
    left, right = np.asarray(actual), np.asarray(expected)
    error = np.abs(left - right)
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise AssertionError(f"ERROR {label}: nonfinite values")
    if not np.all(error <= tolerance * (1 + np.abs(right))):
        raise AssertionError(f"ERROR {label}: max absolute error={error.max():.6g}")


def pack(field: ArrayLike, px: int, py: int) -> np.ndarray:
    """Populate packed halos using global periodic x and clipped y coordinates."""
    interior = np.asarray(field)[2:-2, 2:-2]
    gx, gy = interior.shape
    nx, ny = gx // px, gy // py
    result = np.empty((px * (nx + 4), py * (ny + 4)), dtype=interior.dtype)
    for rx in range(px):
        for ry in range(py):
            x = (rx * nx + np.arange(nx + 4) - 2) % gx
            y = np.clip(ry * ny + np.arange(ny + 4) - 2, 0, gy - 1)
            result[
                rx * (nx + 4) : (rx + 1) * (nx + 4), ry * (ny + 4) : (ry + 1) * (ny + 4)
            ] = interior[np.ix_(x, y)]
    return result


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
            value = jax.device_put(data, sharding)
            tangent = jax.device_put(direction, sharding)
            actual, derivative = jax.jvp(refresh, (value,), (tangent,))
            _, pullback = jax.vjp(refresh, value)
            gradient = pullback(jax.device_put(weights, sharding))[0]
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
    serial, conf, phys = initialize_island(nx, ny, settings_overrides=options)
    x, y = jnp.indices(serial.hIceMean.shape, dtype=dtype)
    serial = replace(
        serial,
        hIceMean=(0.8 + 0.03 * jnp.sin(x + y)) * serial.iceMask,
        Area=0.7 * serial.iceMask,
        uWind=4 + 0.3 * jnp.cos(y),
        vWind=2 + 0.2 * jnp.sin(x + y),
    )
    serial = fill_state_overlap(serial, conf)
    assert np.all(np.asarray(serial.iceMask)[2:-2, 2] == 1)
    assert np.all(np.asarray(serial.iceMask)[2:-2, -3] == 1)
    packed = {
        field.name: pack(getattr(serial, field.name), px, py)
        for field in fields(serial)
    }
    with jax.set_mesh(mesh):
        parallel, local_conf, local_phys = initialize(
            nx // px,
            ny // py,
            mesh=mesh,
            settings_overrides=options,
            state_overrides=packed,
        )
    point = jnp.zeros(2, dtype=dtype)
    spatial_weights = 1 + 0.2 * jnp.sin(
        jnp.arange(nx)[:, None] + 2 * jnp.arange(ny)[None, :]
    )

    def evolve(parameters: jax.Array, sharded: bool) -> State:
        state, settings, physical = (
            (parallel, local_conf, local_phys) if sharded else (serial, conf, phys)
        )
        state = replace(state, hIceMean=state.hIceMean * (1 + parameters[0]))
        for cooling in (80.0, 110.0):
            state = step(state, settings, physical, cooling + 50 * parameters[1])
        return state

    def objective(parameters: jax.Array, sharded: bool) -> jax.Array:
        result = evolve(parameters, sharded)
        value = result.hIceMean + 10 * result.vIce
        physical = remove_halos(value, mesh) if sharded else value[2:-2, 2:-2]
        return jnp.mean(spatial_weights * physical)

    reference = evolve(point, False)
    expected_gradient = jax.grad(lambda p: objective(p, False))(point)
    with jax.set_mesh(mesh):
        replicated = NamedSharding(mesh, P())
        point = jax.device_put(point, replicated)
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
        north_shear = np.concatenate(
            [
                np.asarray(actual.sigma12)[
                    rx * (local_nx + 4) + 2 : rx * (local_nx + 4) + local_nx + 2,
                    -2,
                ]
                for rx in range(px)
            ]
        )
        compare(north_shear, reference.sigma12[2:-2, -2], "north wall shear", tolerance)
        compare(np.asarray(actual.vIce)[:, :3], 0, "south wall velocity", 0)
        compare(np.asarray(actual.vIce)[:, -2:], 0, "north wall velocity", 0)
        function = jax.jit(lambda p: objective(p, True))
        value, pullback = jax.vjp(function, point)
        gradient = pullback(jnp.ones_like(value))[0]
        compare(value, objective(point, False), "coupled scalar objective", tolerance)
        compare(gradient, expected_gradient, "coupled VJP serial/sharded", tolerance)
        for axis in range(2):
            direction = jax.device_put(jnp.eye(2, dtype=dtype)[axis], replicated)
            _, tangent = jax.jvp(function, (point,), (direction,))
            compare(tangent, gradient[axis], f"coupled JVP/VJP axis={axis}", tolerance)
            assert abs(float(tangent)) > 1e-8, "ERROR zero coupled sensitivity"
            delta = 0.01 if dtype == "float32" else 1e-4
            finite_difference = (
                function(point + delta * direction)
                - function(point - delta * direction)
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
def main(backend: str, dtype: str, halos_only: bool) -> None:
    """Exercise all device-grid orientations and both wall-slip choices."""
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices(backend)
    count = len(devices)
    assert count >= 2, "ERROR at least two devices required"
    layouts = [(1, count), (count, 1)]
    if count >= 4 and count % 2 == 0:
        layouts.append((2, count // 2))
    with jax.default_device(devices[0]):
        for px, py in layouts:
            mesh = Mesh(np.asarray(devices, dtype=object).reshape(px, py), ("x", "y"))
            for local_ny in (2, 5):
                check_halos(mesh, local_ny)
        if not halos_only:
            px, py = layouts[-1] if count >= 4 else (1, count)
            mesh = Mesh(np.asarray(devices, dtype=object).reshape(px, py), ("x", "y"))
            for no_slip in (False, True):
                check_coupled(mesh, no_slip, dtype)


if __name__ == "__main__":
    main()
