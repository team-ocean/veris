"""Compare zero-strain wind sensitivities across serial and sharded execution."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from veris import Execution, Experiment, Grid, initialize_experiment
from veris.execution.advance import advance as advance_model
from veris.grid.halos import remove_halos

compiled_advance = jax.jit(
    advance_model, static_argnames=("configuration", "constants", "physics")
)


def check() -> None:
    """Check two evolving steps, JVP/VJP agreement and serial finite differences."""
    count = len(jax.devices())
    shape = (2, 2) if count == 4 else (1, count)
    mesh = jax.make_mesh(shape, ("x", "y"))
    state, conf, phys = initialize_experiment(
        Experiment.builtin("dynamics", grid=Grid(8, 12), settings={"nEVPsteps": 2})
    )
    distributed, dc, dp = initialize_experiment(
        Experiment.builtin(
            "dynamics",
            grid=Grid(8, 12),
            settings={"nEVPsteps": 2},
            execution=Execution(partitions=(mesh.shape["x"], mesh.shape["y"])),
        ),
        mesh=mesh,
    )

    def serial(scale: jax.Array) -> jax.Array:
        current = replace(state, uWind=state.uWind * scale, vWind=state.vWind * scale)
        for _ in range(2):
            current = compiled_advance(current, conf, phys, physics="dynamics")[0]
        return jnp.sum(current.uIce[2:-2, 2:-2] ** 2 + current.vIce[2:-2, 2:-2] ** 2)

    def sharded(scale: jax.Array) -> jax.Array:
        current = replace(
            distributed,
            uWind=distributed.uWind * scale,
            vWind=distributed.vWind * scale,
        )
        for _ in range(2):
            current = compiled_advance(current, dc, dp, physics="dynamics")[0]
        return jnp.sum(
            remove_halos(current.uIce, mesh) ** 2
            + remove_halos(current.vIce, mesh) ** 2
        )

    value = jnp.asarray(1.0)
    sv, sj = jax.jvp(serial, (value,), (value,))
    sg = jax.grad(serial)(value)
    with jax.set_mesh(mesh):
        dv, dj = jax.jvp(sharded, (value,), (value,))
        dg = jax.grad(sharded)(value)
    for label, actual, expected in [
        ("value", dv, sv),
        ("JVP", dj, sj),
        ("VJP", dg, sg),
        ("duality", dg, dj),
    ]:
        assert np.isfinite(actual), f"ERROR nonfinite {label}"
        np.testing.assert_allclose(
            actual, expected, rtol=1e-9, atol=1e-11, err_msg=label
        )
    delta = 1e-3
    fd = (serial(value + delta) - serial(value - delta)) / (2 * delta)
    np.testing.assert_allclose(dg, fd, rtol=3e-5, atol=1e-10)
    print(
        f"zero-strain AD: {count} {jax.default_backend()} devices, evolving JVP/VJP/FD passed"
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    check()
