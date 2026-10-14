"""Load setup definitions and run the existing host Simulation lifecycle.

Python setup files expose build_experiment() -> Experiment; JSON files use the
versioned model codec. This module orchestrates host resources only, without
implementing physical equations. Outputs retain global physical (x, y) fields
and the definition's calendars/schedules. Sharding uses explicit JAX meshes and
the maintained parallel driver's distributed configuration. Python factories
execute trusted user code and must defer device allocation to Simulation.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

from veris import Experiment
from veris.cli.setups import resolve_setup


def load_experiment(source: str | Path) -> Experiment:
    """Load a file or named factory, allowing imports from its setup directory."""
    source = resolve_setup(source).resolve()
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
    source: str | Path,
    *,
    output: Path,
    netcdf: Path | None = None,
    steps: int | None = None,
    float_type: str | None = None,
    backend: str | None = None,
    mesh: tuple[int, int] | None = None,
    io_variables: str | None = None,
    set_values: tuple[str, ...] = (),
) -> None:
    """Load, translate explicit flags and invoke the same public Python runner."""
    import click

    import veris
    from veris.cli.overrides import combine_overrides, parse_overrides

    dedicated: dict[str, Any] = {}
    if steps is not None:
        dedicated["steps"] = steps
    if float_type is not None:
        dedicated.update(
            settings={"dtype": float_type}, constants={"dtype": float_type}
        )
    execution: dict[str, Any] = {}
    if backend is not None:
        execution["backend"] = backend
    if mesh is not None:
        execution.update(partitions=mesh, all_devices=False)
    if execution:
        dedicated["execution"] = execution
    overrides = combine_overrides(parse_overrides(set_values), dedicated)
    try:
        result = veris.run_experiment(
            load_experiment(source),
            overrides=overrides,
            output_path=netcdf,
            snapshot_path=output,
            variables=None if io_variables is None else tuple(io_variables.split(",")),
        )
    except ValueError as error:
        if str(error) == "enabled output requires an output_path":
            raise ValueError("enabled output requires --netcdf PATH") from error
        raise
    assert result.execution_metadata is not None
    if result.execution_metadata["process_index"] == 0:
        click.echo(f"Saved {result.completed_steps} steps to {output}")
