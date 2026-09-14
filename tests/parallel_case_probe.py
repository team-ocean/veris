"""Compare evolving dynamics and gathered physical output on four CPU devices."""

import jax
import numpy as np

from veris.setups import run_dyn, run_parallel


def check_parallel_case() -> None:
    """Ensure rectangular partition interiors equal the serial reference case."""
    mesh = run_parallel.create_mesh((2, 2), "cpu")
    overrides = {"nEVPsteps": 2}
    serial, settings, physical = run_dyn.initialize(
        12, 16, settings_overrides=overrides
    )
    with jax.set_mesh(mesh):
        parallel, local_settings, local_physical = run_dyn.initialize(
            12, 16, mesh=mesh, settings_overrides=overrides
        )
    for _ in range(2):
        serial = run_dyn.compiled_step(serial, settings, physical)
        with jax.set_mesh(mesh):
            parallel = run_dyn.compiled_step(parallel, local_settings, local_physical)
    gathered = run_parallel.gather_output(parallel, mesh)
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
