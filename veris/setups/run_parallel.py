"""Run the reference dynamics case on a local or distributed JAX mesh.

Adapted from ``jax_halo_exchange:run_parallel.py``. Global physical nx/ny are
partitioned by run_dyn.initialize; each device stores its own two-cell halos.
The timed integration compiles scan shapes before executing the trajectory.
Output contains
the nine reference fields after removing every partition's halos, in (x, y)
order. CPU ranks use one device each; local runs may expose several devices via
JAX_NUM_CPU_DEVICES. GPU execution targets the local NVIDIA CUDA devices.
Distributed initialization precedes backend allocation.
"""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import click
import jax

from veris.execution.runtime import create_mesh, distributed_options
from veris.io.cli import (
    output_options,
    output_settings,
    parse_options,
    prepare_output_paths,
)

OUTPUT_FIELDS = (
    "hIceMean",
    "Area",
    "hSnowMean",
    "uIce",
    "vIce",
    "uWind",
    "vWind",
    "uOcean",
    "vOcean",
)


@click.command(help=__doc__)
@click.option("--nx", type=click.IntRange(min=2), default=1024)
@click.option("--ny", type=click.IntRange(min=2))
@click.option("--mesh", nargs=2, type=click.IntRange(min=1), metavar="PX PY")
@click.option("--steps", type=click.IntRange(min=1), default=1000)
@click.option("--evp-steps", type=click.IntRange(min=1), default=120)
@click.option("--backend", type=click.Choice(["cpu", "gpu"]), default="cpu")
@output_options("parallel.nc", OUTPUT_FIELDS)
def cli(**kwargs: Any) -> SimpleNamespace:
    """Parse global physical grid, device topology and integration options."""
    return SimpleNamespace(**kwargs)


def parse_args(argv: list[str] | None = None) -> SimpleNamespace:
    """Validate command arguments before distributed initialization."""
    return parse_options(cli, argv)


def main(argv: list[str] | None = None) -> None:
    """Initialize distributed communication, integrate, gather and shut down."""
    arguments = parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    # Explicit "gpu" platforms expand to every accelerator plugin in JAX;
    # selecting CUDA avoids requiring an unrelated ROCm installation.
    jax.config.update("jax_platforms", "cuda" if arguments.backend == "gpu" else "cpu")
    options = distributed_options(os.environ, arguments.backend)
    started_distributed = False
    try:
        if options is not None:
            if arguments.backend == "cpu":
                jax.config.update("jax_num_cpu_devices", 1)
                jax.config.update("jax_cpu_collectives_implementation", "gloo")
            jax.distributed.initialize(**options)
            started_distributed = True

        from veris import Execution, Experiment, Grid, Simulation

        partitions = tuple(arguments.mesh) if arguments.mesh else None
        mesh = create_mesh(partitions, arguments.backend)
        layout = mesh.shape["x"], mesh.shape["y"]
        definition = Experiment.builtin(
            "dynamics",
            grid=Grid(
                arguments.nx, arguments.nx if arguments.ny is None else arguments.ny
            ),
            settings={"nEVPsteps": arguments.evp_steps},
            execution=Execution(backend=arguments.backend, partitions=layout),
            steps=arguments.steps,
        )
        definition = replace(
            definition,
            output=output_settings(arguments, definition.step_duration.total_seconds()),
        )
        prepare_output_paths(arguments)
        # A one-device topology uses the serial facade; its status still reports
        # the explicit 1x1 CLI topology. Communication remains caller-owned.
        execution_mesh = None if layout == (1, 1) else mesh
        with Simulation.initialize(
            definition, mesh=execution_mesh, output_path=arguments.netcdf
        ) as simulation:
            result = simulation.run()
            simulation.snapshot(
                arguments.output, variables=tuple(arguments.io_variables.split(","))
            )
        compilation, elapsed = result.compilation_seconds, result.integration_seconds
        rank = jax.process_index()
        if rank == 0:
            print(
                f"mesh={mesh.shape['x']}x{mesh.shape['y']} backend={arguments.backend} steps={arguments.steps}"
            )
            print(
                f"compilation={compilation:.3f}s integration={elapsed:.3f}s output={arguments.output}"
            )
    finally:
        if started_distributed:
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
