"""Shared opt-in netCDF options for maintained standalone Veris drivers.

CLI histories default to one sample per model step; the lower-level API
supports independent stream schedules. Final snapshots contain physical cells.
"""


# ruff: noqa: DTZ001 -- model dates intentionally have no timezone.

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import click

from veris.io.calendar import FixedDate, duration_us
from veris.io.configuration import (
    OUTPUT_SETTINGS,
    STREAM_SETTINGS,
    OutputSettings,
    Stream,
)

_Command = TypeVar("_Command", bound=Callable[..., Any])


def output_options(
    default_path: str, variables: tuple[str, ...]
) -> Callable[[_Command], _Command]:
    """Decorate a Click command with netCDF snapshot and history controls."""

    def decorate(command: _Command) -> _Command:
        """Attach common output options while preserving the callback type."""
        options = [
            click.option(
                "--output",
                type=click.Path(path_type=Path),
                default=default_path,
                show_default=True,
                help="Final physical-field netCDF snapshot",
            ),
            click.option(
                "--netcdf",
                type=click.Path(path_type=Path),
                help="Optional sampled netCDF history",
            ),
            click.option("--io-variables", default=",".join(variables)),
            click.option(
                "--sample-seconds",
                type=float,
                help="Sampling interval (default: model timestep)",
            ),
            click.option(
                "--average",
                default=STREAM_SETTINGS["period"].default,
                help="Comma-separated instantaneous,daily,monthly,annual streams",
            ),
            click.option(
                "--calendar",
                type=click.Choice(["gregorian", "360_day", "noleap"]),
                default=OUTPUT_SETTINGS["calendar"].default,
            ),
            click.option(
                "--start-date",
                default=OUTPUT_SETTINGS["start"].default.strftime("%Y-%m-%d"),
                help="Model origin YYYY-MM-DD",
            ),
        ]
        for option in reversed(options):
            command = option(command)
        return command

    return decorate


def parse_options(
    command: click.Command, argv: Sequence[str] | None
) -> SimpleNamespace:
    """Parse Click options while preserving callable driver entry points."""
    try:
        result = command.main(args=argv, standalone_mode=False)
    except click.ClickException as error:
        error.show()
        raise SystemExit(error.exit_code) from error
    if not isinstance(result, SimpleNamespace):
        raise SystemExit(0)
    return result


def output_settings(args: SimpleNamespace, step_seconds: float) -> OutputSettings:
    """Parse host output controls without opening files or allocating arrays."""
    if args.netcdf is not None and args.netcdf.resolve() == args.output.resolve():
        raise ValueError("history and final snapshot require distinct paths")
    step = timedelta(seconds=step_seconds)
    interval = timedelta(
        seconds=step_seconds if args.sample_seconds is None else args.sample_seconds
    )
    if duration_us(interval) <= 0 or duration_us(interval) % duration_us(step):
        raise ValueError(
            "sampling interval must be a positive integer multiple of model timestep"
        )
    year, month, day = (int(part) for part in args.start_date.split("-"))
    start = (
        datetime(year, month, day)
        if args.calendar == "gregorian"
        else FixedDate(year, month, day)
    )
    streams = tuple(
        Stream(period, tuple(args.io_variables.split(",")), interval, period)
        for period in args.average.split(",")
    )
    settings = OutputSettings(
        start=start,
        calendar=args.calendar,
        streams=streams,
        enabled=args.netcdf is not None,
    )
    return settings


def prepare_output_paths(args: SimpleNamespace) -> None:
    """Create parents for validated CLI snapshot/history destinations.

    Drivers call this after validating their Experiment and output settings.
    Existing files are never opened or replaced here; public Simulation keeps
    its strict parent and no-clobber preflight for general API callers.
    """
    for destination in (args.output, args.netcdf):
        if destination is not None:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)


def parameter_namespace(options: Mapping[str, Any]) -> SimpleNamespace:
    """Retain which Click controls were explicitly supplied, including falsy ones."""
    context = click.get_current_context()
    provided = frozenset(
        name
        for name in options
        if context.get_parameter_source(name) != click.core.ParameterSource.DEFAULT
    )
    return SimpleNamespace(**options, _provided=provided)


def driver_definition(name: str, profile: str, args: SimpleNamespace) -> Any:
    """Translate module flags into one Experiment override map without resources."""
    from veris import Experiment
    from veris.cli.overrides import combine_overrides, parse_overrides

    generic = parse_overrides(args.set_values)
    provided = args._provided
    dedicated: dict[str, Any] = {}
    if "steps" in provided:
        dedicated["steps"] = args.steps
    grid = {}
    if "nx" in provided:
        grid["nx"] = args.nx
        if "ny" not in provided and "ny" not in generic.get("grid", {}):
            grid["ny"] = args.nx
    if "ny" in provided:
        grid["ny"] = args.ny
    if grid:
        dedicated["grid"] = grid
    if "evp_steps" in provided:
        dedicated["settings"] = {"nEVPsteps": args.evp_steps}
    execution = {}
    if "backend" in provided:
        execution["backend"] = args.backend
    if "mesh" in provided:
        execution.update(partitions=tuple(args.mesh), all_devices=False)
    if execution:
        dedicated["execution"] = execution
    output: dict[str, Any] = {}
    if "calendar" in provided:
        output["calendar"] = args.calendar
    if "start_date" in provided:
        year, month, day = (int(part) for part in args.start_date.split("-"))
        output["start"] = {"year": year, "month": month, "day": day}
    if output:
        if "enabled" not in generic.get("output", {}):
            output["enabled"] = args.netcdf is not None
        dedicated["output"] = output
    base = Experiment.builtin(name, profile=profile)
    definition = base.with_overrides(combine_overrides(generic, dedicated))
    if output and base.output is None and "streams" not in generic.get("output", {}):
        definition = definition.with_overrides(
            {
                "output": {
                    "streams": (
                        Stream(
                            variables=tuple(args.io_variables.split(",")),
                            sampling_interval=definition.step_duration,
                        ),
                    )
                }
            }
        )
    if {"sample_seconds", "average"} & provided:
        source = definition.output
        interval = (
            definition.step_duration
            if args.sample_seconds is None
            else timedelta(seconds=args.sample_seconds)
        )
        if "average" in provided or source is None:
            streams = tuple(
                Stream(period, tuple(args.io_variables.split(",")), interval, period)
                for period in args.average.split(",")
            )
        else:
            streams = tuple(
                replace(stream, sampling_interval=interval) for stream in source.streams
            )
        # Stream arrays are replaced as whole values; dedicated scheduling flags
        # must not silently replace a grouped --set stream request.
        overrides = combine_overrides(
            generic,
            {
                **dedicated,
                "output": {
                    "enabled": args.netcdf is not None,
                    **output,
                    "streams": streams,
                },
            },
        )
        definition = Experiment.builtin(name, profile=profile).with_overrides(overrides)
        if duration_us(interval) <= 0 or duration_us(interval) % duration_us(
            definition.step_duration
        ):
            raise ValueError(
                "sampling interval must be a positive integer multiple of model timestep"
            )
    if args.netcdf is not None and args.netcdf.resolve() == args.output.resolve():
        raise ValueError("history and final snapshot require distinct paths")
    return definition


def run_driver(definition: Any, args: SimpleNamespace) -> Any:
    """Invoke the public runner with module snapshot/history destinations."""
    import veris

    return veris.run_experiment(
        definition,
        output_path=args.netcdf,
        snapshot_path=args.output,
        variables=tuple(args.io_variables.split(",")),
    )


def console_main(callback: Callable[[], Any]) -> None:
    """Report module-console host errors concisely without changing Python APIs."""
    try:
        callback()
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        click.ClickException(str(error)).show()
        raise SystemExit(1) from error
