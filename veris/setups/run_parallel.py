"""Run the reference dynamics case on a local or distributed JAX mesh.

Adapted from ``jax_halo_exchange:run_parallel.py``. Global physical nx/ny are
partitioned by run_dyn.initialize; each device stores its own two-cell halos.
The timed integration discards warmup scans for every executed chunk length.
Output contains
the nine reference fields after removing every partition's halos, in (x, y)
order. CPU ranks use one device each; local runs may expose several devices via
JAX_NUM_CPU_DEVICES. GPU execution targets the local NVIDIA CUDA devices.
Distributed initialization precedes backend allocation.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import jax
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from veris._typing import State
from veris.integration_output import output_callbacks, run_timed
from veris.io.cli import make_output, output_options, parse_options, save_final
from veris.io.distributed import distributed_collector
from veris.io.storage import selected_fields, write_snapshot

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


def distributed_options(
    environment: Mapping[str, str], backend: str
) -> dict[str, Any] | None:
    """Resolve explicit JAX rank variables or a multi-task Slurm launch.

    Supply JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES, and JAX_PROCESS_ID
    together outside Slurm. GPU processes may select comma-separated device
    indices with JAX_LOCAL_DEVICE_IDS; the default is their Slurm local rank,
    or zero outside Slurm. A Slurm allocation without task ranks is local.
    """
    names = ("JAX_COORDINATOR_ADDRESS", "JAX_NUM_PROCESSES", "JAX_PROCESS_ID")
    present = [name in environment for name in names]
    if any(present):
        if not all(present):
            raise ValueError(
                "explicit coordinator/count/process variables must be set together"
            )
        count, rank = int(environment[names[1]]), int(environment[names[2]])
        if count < 1 or not 0 <= rank < count:
            raise ValueError(
                "process count must be positive and process ID within its range"
            )
        local_ids = [0]
        if backend == "gpu":
            local_ids = [
                int(value)
                for value in environment.get(
                    "JAX_LOCAL_DEVICE_IDS", environment.get("SLURM_LOCALID", "0")
                ).split(",")
            ]
            if any(index < 0 for index in local_ids):
                raise ValueError("local device indices must be nonnegative")
        return {
            "coordinator_address": environment[names[0]],
            "num_processes": count,
            "process_id": rank,
            "local_device_ids": local_ids,
        }
    if "SLURM_PROCID" in environment and int(environment.get("SLURM_NTASKS", "1")) > 1:
        result: dict[str, Any] = {"cluster_detection_method": "slurm"}
        if backend == "cpu":
            result["local_device_ids"] = [0]
        return result
    return None


def create_mesh(partitions: tuple[int, int] | None, backend: str) -> Mesh:
    """Create x/y mesh using all devices of the requested execution backend."""
    devices = jax.devices(backend)
    shape = partitions if partitions is not None else (1, len(devices))
    if min(shape) < 1 or shape[0] * shape[1] != len(devices):
        raise ValueError(
            f"mesh {shape} must contain all {len(devices)} available {backend} devices"
        )
    return jax.make_mesh(shape, ("x", "y"), devices=devices)


def remove_halos(array: jax.Array, mesh: Mesh) -> jax.Array:
    """Trim two boundary cells from each shard, retaining global physical order."""

    @partial(jax.shard_map, mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y"))
    def interior(local: jax.Array) -> jax.Array:
        return local[2:-2, 2:-2]

    # The explicit shard_map mesh also works while JVP/VJP/JIT trace callers.
    # Entering a new set_mesh context here would reject those transformations.
    return interior(array)


def gather_output(state: State, mesh: Mesh) -> dict[str, np.ndarray]:
    """Collect only the nine physical reference fields on every participating host."""
    from jax.experimental import multihost_utils

    return {
        name: np.asarray(
            multihost_utils.process_allgather(
                remove_halos(getattr(state, name), mesh), tiled=True
            )
        )
        for name in OUTPUT_FIELDS
    }


def save_output(
    path: Path, fields: Mapping[str, np.ndarray], *, process_index: int
) -> None:
    """Reject nonfinite physical fields and write netCDF on process zero."""
    if process_index == 0:
        for name, field in fields.items():
            if not np.all(np.isfinite(field)):
                raise ValueError(f"nonfinite output in {name}")
        write_snapshot(path, fields, collector=selected_fields)


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

        from veris.setups import run_dyn

        partitions = tuple(arguments.mesh) if arguments.mesh else None
        mesh = create_mesh(partitions, arguments.backend)
        with jax.set_mesh(mesh):
            state, settings, physical = run_dyn.initialize(
                arguments.nx,
                arguments.ny,
                mesh=mesh,
                settings_overrides={"nEVPsteps": arguments.evp_steps},
            )
            advance = partial(run_dyn.compiled_step, conf=settings, phys=physical)
            collect = distributed_collector(mesh)
            with make_output(
                arguments, settings.deltatDyn, collector=collect
            ) as output:
                observe, select = output_callbacks(output, settings.deltatDyn)
                state, warmup, elapsed = run_timed(
                    state,
                    advance,
                    arguments.steps,
                    observe=observe,
                    select=select,
                )
            save_final(
                arguments,
                state,
                timedelta(seconds=arguments.steps * settings.deltatDyn),
                output.settings,
                settings,
                physical,
                collector=collect,
            )
        rank = jax.process_index()
        if rank == 0:
            print(
                f"mesh={mesh.shape['x']}x{mesh.shape['y']} backend={arguments.backend} steps={arguments.steps}"
            )
            print(
                f"warmup={warmup:.3f}s integration={elapsed:.3f}s output={arguments.output}"
            )
    finally:
        if started_distributed:
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
