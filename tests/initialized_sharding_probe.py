"""Four-CPU full-State initialization, coupled-step and forcing-AD comparison.

The serial artificial island spans partition boundaries. Independent modular
indexing packs each partition's interior plus halos from one global serial
field; no production halo exchange is used to construct expected values.
"""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from veris.initialization import initialize
from veris.setup.artificial import compiled_step, step
from veris.setup.artificial import initialize as initialize_artificial
from veris.state import State
from veris.variables import VARIABLES


def pack(field: jax.Array, px: int, py: int) -> np.ndarray:
    """Index the unpartitioned interior periodically to populate every halo."""
    interior = np.asarray(field)[2:-2, 2:-2]
    gx, gy = interior.shape
    nx, ny = gx // px, gy // py
    result = np.empty((px * (nx + 4), py * (ny + 4)), dtype=interior.dtype)
    for rank_x in range(px):
        for rank_y in range(py):
            x = (rank_x * nx + np.arange(nx + 4) - 2) % gx
            y = (rank_y * ny + np.arange(ny + 4) - 2) % gy
            result[
                rank_x * (nx + 4) : (rank_x + 1) * (nx + 4),
                rank_y * (ny + 4) : (rank_y + 1) * (ny + 4),
            ] = interior[np.ix_(x, y)]
    return result


def assert_close(actual: object, expected: object, label: str) -> None:
    """Report one aggregate numerical discrepancy, never entire arrays."""
    actual_array, expected_array = np.asarray(actual), np.asarray(expected)
    error = np.abs(actual_array - expected_array)
    tolerance = 1e-10 + 1e-10 * np.abs(expected_array)
    if not np.all(error <= tolerance):
        location = np.unravel_index(np.argmax(error), error.shape)
        raise AssertionError(
            f"ERROR {label}: max absolute difference={error[location]:.6g} at {location}"
        )


def check_initialized_step() -> None:
    """Compare initialized distributed State, coupled output, JVP and VJP."""
    px, py, nx, ny = 2, 2, 4, 5
    mesh = jax.make_mesh((px, py), ("x", "y"))
    defaults, settings, _ = initialize(nx, ny, mesh=mesh)
    assert settings.use_sharding and settings.nx == nx and settings.ny == ny
    assert len(jax.tree.leaves(defaults)) == 70
    for field in fields(defaults):
        array = getattr(defaults, field.name)
        assert array.shape == (px * (nx + 4), py * (ny + 4))
        assert isinstance(array.sharding, NamedSharding)
        assert array.sharding.spec == P("x", "y")
        assert len(array.addressable_shards) == 4
        assert array.dtype == np.dtype(VARIABLES[field.name].dtype)
        assert_close(array, VARIABLES[field.name].default, f"default {field.name}")

    serial, serial_settings, physical = initialize_artificial(px * nx, py * ny)
    serial_settings = replace(serial_settings, nEVPsteps=2)
    packed = {
        field.name: pack(getattr(serial, field.name), px, py)
        for field in fields(serial)
    }
    distributed, settings, distributed_physical = initialize(
        nx,
        ny,
        mesh=mesh,
        settings_overrides={
            "deltatDyn": serial_settings.deltatDyn,
            "deltatTherm": serial_settings.deltatTherm,
            "nEVPsteps": 2,
        },
        state_overrides=packed,
    )
    assert physical == distributed_physical
    for field in fields(distributed):
        assert_close(
            getattr(distributed, field.name),
            packed[field.name],
            f"override {field.name}",
        )

    expected = step(serial, serial_settings, physical, cooling=100.0)
    with jax.set_mesh(mesh):
        actual = step(distributed, settings, distributed_physical, cooling=100.0)
    jax.block_until_ready(actual)
    for field in fields(actual):
        assert_close(
            getattr(actual, field.name),
            pack(getattr(expected, field.name), px, py),
            field.name,
        )

    # Whole-step JIT must preserve mesh dispatch and every initialized leaf,
    # not only the eager Python composition of individually compiled kernels.
    with jax.set_mesh(mesh):
        compiled = compiled_step(
            distributed, settings, distributed_physical, cooling=100.0
        )
    jax.block_until_ready(compiled)
    for field in fields(compiled):
        assert_close(
            getattr(compiled, field.name),
            pack(getattr(expected, field.name), px, py),
            f"compiled {field.name}",
        )

    # Count each physical cell once, excluding duplicated local halo values.
    weights = np.zeros(distributed.hIceMean.shape)
    for rank_x in range(px):
        for rank_y in range(py):
            weights[
                rank_x * (nx + 4) + 2 : rank_x * (nx + 4) + nx + 2,
                rank_y * (ny + 4) + 2 : rank_y * (ny + 4) + ny + 2,
            ] = 1
    weight_array = jax.device_put(weights, NamedSharding(mesh, P("x", "y")))

    def serial_total(cooling: jax.Array) -> jax.Array:
        result: State = step(serial, serial_settings, physical, cooling)
        return jnp.sum(result.hIceMean[2:-2, 2:-2])

    def distributed_total(cooling: jax.Array) -> jax.Array:
        result: State = step(distributed, settings, distributed_physical, cooling)
        return jnp.sum(result.hIceMean * weight_array)

    cooling, tangent = jnp.asarray(100.0), jnp.asarray(1.0)
    serial_value, serial_jvp = jax.jvp(serial_total, (cooling,), (tangent,))
    serial_vjp = jax.grad(serial_total)(cooling)
    with jax.set_mesh(mesh):
        value, jvp = jax.jvp(distributed_total, (cooling,), (tangent,))
        vjp = jax.grad(distributed_total)(cooling)
    assert_close(value, serial_value, "coupled objective")
    assert_close(jvp, serial_jvp, "forcing JVP")
    assert_close(vjp, serial_vjp, "forcing VJP")
    assert_close(jvp, vjp, "forward/reverse forcing derivative")
    delta = 1e-2
    finite_difference = (
        serial_total(cooling + delta) - serial_total(cooling - delta)
    ) / (2 * delta)
    assert float(jvp) > 0, "ERROR cooling sensitivity is not positive"
    assert_close(jvp, finite_difference, "forcing central difference")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    assert len(jax.devices()) == 4, "ERROR expected four CPU devices"
    check_initialized_step()
    print("initialized State: coupled step, JVP and VJP passed on four CPU devices")
