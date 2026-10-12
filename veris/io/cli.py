"""Shared opt-in netCDF options for maintained standalone Veris drivers.

CLI histories default to one sample per model step; the lower-level API
supports independent stream schedules. Final snapshots contain physical cells.
"""


# ruff: noqa: DTZ001 -- model dates intentionally have no timezone.

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import click
import numpy as np

from veris.io.calendar import FixedDate, duration_us
from veris.io.configuration import OutputSettings, Stream
from veris.io.output import OutputManager
from veris.io.storage import ArrayFields, Collector, storage_fields, write_snapshot
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State

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
                "--final-netcdf",
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
                default="instantaneous",
                help="Comma-separated instantaneous,daily,monthly,annual streams",
            ),
            click.option(
                "--calendar",
                type=click.Choice(["gregorian", "360_day", "noleap"]),
                default="gregorian",
            ),
            click.option(
                "--start-date", default="2000-01-01", help="Model origin YYYY-MM-DD"
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


def make_output(
    args: SimpleNamespace, step_seconds: float, *, collector: Collector | None = None
) -> OutputManager:
    """Create a lazy manager using the shared settings-only parser."""
    return OutputManager(
        args.netcdf or ".", output_settings(args, step_seconds), collector=collector
    )


def prepare_output_paths(args: SimpleNamespace) -> None:
    """Create parents for validated CLI snapshot/history destinations.

    Drivers call this after validating their Experiment and output settings.
    Existing files are never opened or replaced here; public Simulation keeps
    its strict parent and no-clobber preflight for general API callers.
    """
    for destination in (args.output, args.netcdf):
        if destination is not None:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)


def save_final(
    args: SimpleNamespace,
    state: State,
    elapsed: timedelta,
    settings: OutputSettings,
    conf: Configuration,
    phys: PhysicalConstants,
    *,
    collector: Collector | None = None,
) -> None:
    """Write the concrete final State independently of whether history was enabled."""

    def collect_finite(
        source: State | ArrayFields, names: tuple[str, ...]
    ) -> ArrayFields | None:
        """Validate physical output on the writing rank before opening a file."""
        values = (collector or storage_fields)(source, names)
        if values is not None:
            for name, value in values.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"nonfinite output in {name}")
        return values

    write_snapshot(
        args.output,
        state,
        elapsed=elapsed,
        start=settings.start,
        calendar=settings.calendar,
        variables=tuple(args.io_variables.split(",")),
        conf=conf,
        phys=phys,
        collector=collect_finite,
    )
