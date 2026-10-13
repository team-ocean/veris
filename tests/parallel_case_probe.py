"""Compare evolving dynamics and gathered physical output on four CPU devices."""

import jax
import numpy as np

from veris import Execution, Experiment, Grid, initialize_experiment
from veris.execution.advance import advance
from veris.execution.runtime import create_mesh
from veris.io.distributed import distributed_collector
from veris.setups import run_parallel


def check_parallel_case() -> None:
    """Ensure rectangular partition interiors equal the serial reference case."""
    mesh = create_mesh((2, 2), "cpu")
    overrides = {"nEVPsteps": 2}
    serial, settings, physical = initialize_experiment(
        Experiment.builtin("dynamics", grid=Grid(12, 16), settings=overrides)
    )
    with jax.set_mesh(mesh):
        parallel, local_settings, local_physical = initialize_experiment(
            Experiment.builtin(
                "dynamics",
                grid=Grid(12, 16),
                settings=overrides,
                execution=Execution(partitions=(mesh.shape["x"], mesh.shape["y"])),
            ),
            mesh=mesh,
        )
    compiled = jax.jit(
        advance, static_argnames=("configuration", "constants", "physics")
    )
    for _ in range(2):
        serial = compiled(serial, settings, physical, physics="dynamics")[0]
        with jax.set_mesh(mesh):
            parallel = compiled(
                parallel, local_settings, local_physical, physics="dynamics"
            )[0]
    gathered = distributed_collector(mesh)(parallel, run_parallel.OUTPUT_FIELDS)
    assert gathered is not None
    for name, actual in gathered.items():
        expected = np.asarray(getattr(serial, name))[2:-2, 2:-2]
        assert actual.shape == (12, 16), f"ERROR {name} gathered shape"
        error = np.abs(actual - expected)
        assert np.all(error <= 1e-10 + 1e-10 * np.abs(expected)), (
            f"ERROR {name}: max absolute difference={error.max():.6g}"
        )
    print("parallel dynamics: four CPU devices match serial output")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    check_parallel_case()
