"""Direct pure advance forcing, closed-wall and AD checks on two real CPUs.

Run in a fresh process configured for two CPU devices. Compare canonical
transitions with direct numerical kernel composition and separately frozen
scientific references. Forcing halos use explicit periodic/clipped global
coordinates. Both initial States come through the public experiment initializer.
"""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

import veris
from veris.execution.runtime import create_mesh
from veris.grid.halos import fill_state_overlap, remove_halos
from veris.model.diagnostics import Diagnostics
from veris.model.types import State
from veris.physics.dynamics.driver import dynamics_transport

NX, NY = 6, 4
DTYPE = "float64"
OPTIONS = {
    "dtype": DTYPE,
    "enable_cyclic_y": False,
    "nEVPsteps": 2,
    "useAdaptiveEVP": False,
}


def compare(
    actual: ArrayLike,
    expected: ArrayLike,
    label: str,
    tolerance: float = 2e-10,
) -> None:
    """Report finite aggregate errors while avoiding array dumps."""
    left, right = np.asarray(actual), np.asarray(expected)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(
            f"ERROR {label}: metadata {left.shape}/{left.dtype} "
            f"!= {right.shape}/{right.dtype}"
        )
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise AssertionError(f"ERROR {label}: nonfinite values")
    error = np.abs(left - right)
    if not np.all(error <= tolerance * (1 + np.abs(right))):
        raise AssertionError(f"ERROR {label}: max absolute error={error.max():.6g}")


def prepare(state: State) -> State:
    """Give both initializers identical positive ice and nonzero velocity fields."""
    return replace(
        state,
        hIceMean=(0.7 + 0.02 * jnp.sin(state.uWind)) * state.iceMask,
        hSnowMean=0.02 * state.iceMask,
        Area=0.72 * state.iceMask,
        uIce=(0.04 + 0.003 * jnp.cos(state.uWind)) * state.iceMaskU,
        vIce=(0.025 + 0.002 * jnp.sin(state.vWind)) * state.iceMaskV,
        sigma1=(0.15 + 0.01 * jnp.sin(state.uWind)) * state.iceMask,
        sigma2=(0.04 + 0.01 * jnp.cos(state.vWind)) * state.iceMask,
        sigma12=0.005 * jnp.sin(state.uWind + state.vWind) * state.iceMask,
    )


def forcing(parameters: jax.Array) -> dict[str, jax.Array]:
    """Use two differentiable spatial fields and one replicated wind component."""
    x, y = jnp.indices((NX, NY), dtype=DTYPE)
    return {
        "uWind": 4 + 0.3 * x + 0.1 * y + parameters[0] * (0.4 + 0.1 * jnp.cos(y)),
        "vOcean": 0.01
        + 0.002 * x
        + 0.001 * y
        + parameters[1] * (0.015 + 0.003 * jnp.sin(x + y)),
        "vWind": jnp.asarray(1.3, dtype=DTYPE),
    }


def prescribed_serial(state: State, inputs: dict[str, jax.Array]) -> State:
    """Independently populate serial forcing halos before the direct kernel reference."""
    # x is periodic; atmospheric y halos copy the nearest physical cell.
    wind = jnp.pad(inputs["uWind"], ((2, 2), (0, 0)), mode="wrap")
    wind = jnp.pad(wind, ((0, 0), (2, 2)), mode="edge")
    # The owned south normal face and exterior normal halos are zero.
    ocean = inputs["vOcean"].at[:, 0].set(0)
    ocean = jnp.pad(ocean, ((2, 2), (0, 0)), mode="wrap")
    ocean = jnp.pad(ocean, ((0, 0), (2, 2)), mode="constant")
    other_wind = jnp.full_like(state.vWind, inputs["vWind"])
    speed = (
        jnp.full_like(state.wSpeed, inputs["wSpeed"])
        if "wSpeed" in inputs
        else jnp.sqrt(wind**2 + other_wind**2)
    )
    return replace(state, uWind=wind, vWind=other_wind, wSpeed=speed, vOcean=ocean)


def compare_records(
    actual: tuple[State, Diagnostics],
    expected: tuple[State, Diagnostics],
    mesh: Mesh,
    label: str,
) -> None:
    """Compare every numerical State and separate Diagnostics physical leaf."""
    for record, reference in zip(actual, expected, strict=True):
        for item in fields(reference):
            compare(
                remove_halos(getattr(record, item.name), mesh),
                getattr(reference, item.name)[2:-2, 2:-2],
                f"{label} {type(reference).__name__}.{item.name}",
            )


def check_forcing_halos(state: State, inputs: dict[str, jax.Array], mesh: Mesh) -> None:
    """Check each local padded forcing field by independent global coordinates."""
    local_nx, local_ny = NX // mesh.shape["x"], NY // mesh.shape["y"]
    physical_wind, physical_ocean = (
        np.asarray(inputs["uWind"]),
        np.asarray(inputs["vOcean"]),
    )
    scalar_wind = float(inputs["vWind"])
    for shard in state.uWind.addressable_shards:
        sx, sy = shard.index
        rx, ry = (sx.start or 0) // (local_nx + 4), (sy.start or 0) // (local_ny + 4)
        x = (rx * local_nx + np.arange(local_nx + 4) - 2) % NX
        y = ry * local_ny + np.arange(local_ny + 4) - 2
        clipped = np.clip(y, 0, NY - 1)
        expected_wind = physical_wind[np.ix_(x, clipped)]
        expected_ocean = physical_ocean[np.ix_(x, clipped)].copy()
        expected_ocean[:, (y <= 0) | (y >= NY)] = 0
        if "wSpeed" in inputs:
            expected_speed = np.full_like(expected_wind, float(inputs["wSpeed"]))
        else:
            expected_speed = np.sqrt(expected_wind**2 + scalar_wind**2)
        for name, expected in (
            ("uWind", expected_wind),
            ("vOcean", expected_ocean),
            ("wSpeed", expected_speed),
        ):
            actual_shard = next(
                part
                for part in getattr(state, name).addressable_shards
                if part.index == shard.index
            )
            compare(
                actual_shard.data,
                expected,
                f"{mesh.shape['x']}x{mesh.shape['y']} rank {rx},{ry} {name} halos",
            )


def check_orientation(partitions: tuple[int, int]) -> None:
    """Check eager/JIT fields, independent forcing halos, and nonzero pure AD."""
    mesh = create_mesh(partitions, "cpu")
    definition = veris.Experiment.builtin(
        "dynamics", grid=veris.Grid(NX, NY), settings=OPTIONS
    )
    serial, conf, phys = veris.initialize_experiment(definition)
    serial = fill_state_overlap(prepare(serial), conf)
    with jax.set_mesh(mesh):
        parallel, local_conf, local_phys = veris.initialize_experiment(
            replace(
                definition,
                execution=veris.Execution(backend="cpu", partitions=partitions),
            ),
            mesh=mesh,
        )
        parallel = fill_state_overlap(prepare(parallel), local_conf)

    def reference(inputs: dict[str, jax.Array]) -> tuple[State, Diagnostics]:
        state, stress_u, stress_v = dynamics_transport(
            prescribed_serial(serial, inputs), conf, phys
        )
        zeros = jnp.zeros_like(state.Area)
        diagnostic = Diagnostics(zeros, stress_u, stress_v, zeros, zeros, zeros)
        return fill_state_overlap(state, conf), fill_state_overlap(diagnostic, conf)

    def transition(
        current: State, inputs: dict[str, jax.Array]
    ) -> tuple[State, Diagnostics]:
        return veris.advance(
            current, local_conf, local_phys, inputs, physics="dynamics"
        )

    point = jnp.zeros(2, dtype=DTYPE)
    inputs = forcing(point)
    expected = reference(inputs)
    explicit_speed = {**inputs, "wSpeed": jnp.asarray(7.0, dtype=DTYPE)}
    expected_explicit = reference(explicit_speed)
    weights = 1 + 0.2 * jnp.sin(jnp.arange(NX)[:, None] + 2 * jnp.arange(NY)[None, :])

    def objective(parameters: jax.Array, sharded: bool) -> jax.Array:
        result, diagnostics = (
            transition(parallel, forcing(parameters))
            if sharded
            else reference(forcing(parameters))
        )
        value = (
            result.hIceMean
            + 10 * result.uIce
            + 7 * result.vIce
            + diagnostics.OceanStressU
            + diagnostics.OceanStressV
        )
        value = remove_halos(value, mesh) if sharded else value[2:-2, 2:-2]
        return jnp.mean(weights * value)

    expected_value = objective(point, False)
    expected_gradient = jax.grad(lambda p: objective(p, False))(point)
    with jax.set_mesh(mesh):
        physical_sharding, replicated = (
            NamedSharding(mesh, P("x", "y")),
            NamedSharding(mesh, P()),
        )

        def place(inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
            return {
                name: jax.device_put(
                    value, replicated if value.ndim == 0 else physical_sharding
                )
                for name, value in inputs.items()
            }

        inputs = place(inputs)
        actual = transition(parallel, inputs)
        compare_records(actual, expected, mesh, f"{partitions} eager")
        check_forcing_halos(actual[0], inputs, mesh)
        compiled = jax.jit(transition)
        actual = compiled(parallel, inputs)
        compare_records(actual, expected, mesh, f"{partitions} JIT")
        check_forcing_halos(actual[0], inputs, mesh)
        explicit_speed = place(explicit_speed)
        explicit = compiled(parallel, explicit_speed)
        compare_records(
            explicit, expected_explicit, mesh, f"{partitions} explicit speed"
        )
        check_forcing_halos(explicit[0], explicit_speed, mesh)

        point = jax.device_put(point, replicated)
        function = jax.jit(lambda p: objective(p, True))
        gradient = jax.jit(jax.grad(function))(point)
        value, pullback = jax.vjp(function, point)
        compare(value, expected_value, f"{partitions} scalar objective")
        compare(gradient, expected_gradient, f"{partitions} grad serial/sharded")
        compare(pullback(jnp.ones_like(value))[0], gradient, f"{partitions} VJP/grad")
        for axis, name in enumerate(("uWind", "vOcean")):
            tangent = jax.device_put(jnp.eye(2, dtype=DTYPE)[axis], replicated)
            _, derivative = jax.jvp(function, (point,), (tangent,))
            compare(derivative, gradient[axis], f"{partitions} {name} JVP/grad")
            assert abs(float(derivative)) > 1e-9, f"ERROR zero {name} sensitivity"
            epsilon = 1e-4
            difference = (
                function(point + epsilon * tangent)
                - function(point - epsilon * tangent)
            ) / (2 * epsilon)
            left, right = float(derivative), float(difference)
            assert np.isfinite(right) and abs(left - right) <= 2e-8 + 2e-4 * abs(
                right
            ), (
                f"ERROR {partitions} {name} finite difference: AD={left:.6g}, FD={right:.6g}"
            )
    print(
        f"pure advance {partitions}: all fields, local forcing halos, JIT and AD pass"
    )


def main() -> None:
    """Run both two-device orientations with explicit backend/precision setup."""
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices("cpu")
    assert len(devices) == 2, "ERROR probe requires exactly two CPU devices"
    with jax.default_device(devices[0]):
        for partitions in ((2, 1), (1, 2)):
            check_orientation(partitions)
    print("pure advance two-CPU checks passed")


if __name__ == "__main__":
    main()
