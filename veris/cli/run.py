"""Load setup definitions and run the existing host Simulation lifecycle.

Python setup files expose build_experiment() -> Experiment; JSON files use the
versioned model codec. This module orchestrates host resources only, without
implementing physical equations. Outputs retain global physical (x, y) fields
and the definition's calendars/schedules. Sharding uses explicit JAX meshes and
the maintained parallel driver's distributed configuration. Python factories
execute trusted user code and must defer device allocation to Simulation.
"""

from __future__ import annotations

import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import jax

from veris import Execution, Experiment, Simulation
from veris.io import OutputSettings, Stream
from veris.model.variables import VARIABLES


def load_experiment(source: Path) -> Experiment:
    """Load a JSON definition or invoke a Python factory with sibling imports."""
    source = source.resolve()
    if source.suffix == ".json":
        return Experiment.from_file(source)
    if source.suffix != ".py":
        raise ValueError("setup file must have a .py or .json extension")
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(source.parent))
        namespace = runpy.run_path(str(source), run_name="__veris_setup__")
        factory = namespace.get("build_experiment")
        if not callable(factory):
            raise TypeError("Python setup must define build_experiment()")
        definition = factory()
        if not isinstance(definition, Experiment):
            raise TypeError("build_experiment() must return an Experiment")
        return definition
    except Exception as error:
        raise ValueError(f"{source}: {error}") from error
    finally:
        sys.path[:] = previous_path


def run_experiment(
    source: Path,
    *,
    output: Path,
    netcdf: Path | None = None,
    steps: int | None = None,
    float_type: str | None = None,
    backend: str | None = None,
    mesh: tuple[int, int] | None = None,
    io_variables: str | None = None,
) -> None:
    """Validate overrides/paths, own execution resources, and save physical output.

    Omitted controls preserve the file's definition. A requested history uses
    the definition's output settings or a per-step instantaneous stream. Final
    snapshots default to all native fields. Existing destinations never change.
    """
    definition = load_experiment(source)
    settings, constants = dict(definition.settings), dict(definition.constants)
    if float_type is not None:
        settings["dtype"] = float_type
        constants["dtype"] = float_type
    execution = Execution(
        backend=definition.execution.backend if backend is None else backend,
        partitions=definition.execution.partitions if mesh is None else mesh,
    )
    definition = replace(
        definition,
        settings=settings,
        constants=constants,
        execution=execution,
        steps=definition.steps if steps is None else steps,
    )
    variables = (
        tuple(VARIABLES) if io_variables is None else tuple(io_variables.split(","))
    )
    if (
        not variables
        or len(set(variables)) != len(variables)
        or any(name not in VARIABLES for name in variables)
    ):
        raise ValueError("variables must be a nonempty unique selection from VARIABLES")
    if netcdf is not None and netcdf.resolve() == output.resolve():
        raise ValueError("history and final snapshot require distinct paths")
    for path in (output, netcdf):
        if path is not None and (path.exists() or path.is_symlink()):
            raise FileExistsError(f"output already exists: {path}")
    if netcdf is not None:
        schedule = definition.output
        if schedule is None:
            schedule = OutputSettings(
                streams=(
                    Stream(
                        variables=variables, sampling_interval=definition.step_duration
                    ),
                )
            )
        definition = replace(definition, output=replace(schedule, enabled=True))
    elif definition.output is not None and definition.output.enabled:
        raise ValueError("enabled output in setup requires --netcdf PATH")

    # Scheduled reductions use float64 even with float32 numerical fields.
    # Runtime policy is explicit here; help/discovery/import never changes it.
    jax.config.update("jax_enable_x64", True)
    from veris.setups.run_parallel import create_mesh, distributed_options

    selected_backend = execution.backend
    options = distributed_options(os.environ, selected_backend or "cpu")
    started_distributed = False
    try:
        if options is not None:
            if selected_backend is None:
                raise ValueError(
                    "distributed execution requires an explicit cpu or gpu backend"
                )
            if execution.partitions == (1, 1):
                raise ValueError("distributed execution requires a partitioned mesh")
            if selected_backend == "cpu":
                jax.config.update("jax_num_cpu_devices", 1)
                jax.config.update("jax_cpu_collectives_implementation", "gloo")
            jax.distributed.initialize(**options)
            started_distributed = True
        execution_mesh = (
            None
            if execution.partitions == (1, 1)
            else create_mesh(
                execution.partitions, selected_backend or jax.default_backend()
            )
        )
        for path in (output, netcdf):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        with Simulation.initialize(
            definition, mesh=execution_mesh, output_path=netcdf
        ) as simulation:
            result = simulation.run()
            simulation.snapshot(output, variables=variables)
        if jax.process_index() == 0:
            import click

            click.echo(f"Saved {result.completed_steps} steps to {output}")
    finally:
        if started_distributed:
            jax.distributed.shutdown()
