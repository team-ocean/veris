"""Public host runner around validated resources and the existing Simulation.

The runner implements no sea-ice equations. It preflights immutable definitions,
calendars and destinations before acquiring resources; then it initializes,
executes, synchronizes and writes requested physical output through Simulation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax

from veris.execution.experiment_initialization import _validate_inputs
from veris.execution.runtime import execution_resources
from veris.execution.simulation import RunResult, Simulation
from veris.io.configuration import OutputSettings, Stream
from veris.io.guard import require_host
from veris.io.schedule import iter_segments
from veris.model.experiment import Experiment
from veris.model.variables import VARIABLES


def _destination(path: str | Path | None) -> Path | None:
    """Validate no-clobber paths, allowing parents created only after preflight."""
    if path is None:
        return None
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output already exists: {destination}")
    parent = destination.parent
    while not parent.exists():
        if parent.is_symlink():
            raise FileNotFoundError(f"output parent is a dangling symlink: {parent}")
        parent = parent.parent
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    return destination


def run_experiment(
    experiment: Experiment,
    *,
    overrides: Mapping[str, Any] | None = None,
    mesh: Any = None,
    output_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    variables: Sequence[str] | None = None,
) -> RunResult:
    """Execute an experiment and return synchronized State, diagnostics and clock.

    Explicit grouped overrides resolve before preflight. A requested history
    enables the definition's schedule or a per-step instantaneous default.
    Snapshot/history parents are created after successful input/resource checks;
    existing destinations are never replaced. Owned services and writers close
    on success or failure, while supplied meshes/services remain borrowed.
    """
    require_host()
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    definition = (
        experiment if overrides is None else experiment.with_overrides(overrides)
    )
    names = tuple(VARIABLES) if variables is None else tuple(variables)
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in VARIABLES for name in names)
    ):
        raise ValueError("variables must be a nonempty unique selection from VARIABLES")
    history, snapshot = _destination(output_path), _destination(snapshot_path)
    if (
        history is not None
        and snapshot is not None
        and history.resolve() == snapshot.resolve()
    ):
        raise ValueError("history and final snapshot require distinct paths")
    if history is not None:
        schedule = definition.output
        if schedule is None:
            schedule = OutputSettings(
                streams=(
                    Stream(variables=names, sampling_interval=definition.step_duration),
                )
            )
        definition = definition.with_overrides(
            {"output": replace(schedule, enabled=True)}
        )
    if definition.output is not None and definition.output.enabled:
        if history is None:
            raise ValueError("enabled output requires an output_path")
        iter_segments(
            definition.output,
            definition.step_duration.total_seconds(),
            definition.steps,
        )
    _validate_inputs(definition)
    with execution_resources(definition, mesh=mesh) as resources:
        for destination in (history, snapshot):
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            jax.default_device(resources.device),
            Simulation.initialize(
                resources.experiment, mesh=resources.mesh, output_path=history
            ) as simulation,
        ):
            result = simulation.run()
            jax.block_until_ready((result.state, result.diagnostics))
            if snapshot is not None:
                simulation.snapshot(snapshot, variables=names)
        return replace(result, execution_metadata=resources.metadata)


__all__ = ["run_experiment"]
