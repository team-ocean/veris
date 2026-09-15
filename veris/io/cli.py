"""Shared opt-in netCDF options for maintained standalone Veris drivers.

The legacy NPZ outputs remain available. CLI histories default to one sample per
model step; the lower-level API supports independent stream schedules.
"""


# ruff: noqa: DTZ001 -- model dates intentionally have no timezone.

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from veris._typing import State
from veris.configuration import Configuration
from veris.io.calendar import FixedDate, duration_us
from veris.io.configuration import OutputSettings, Stream
from veris.io.output import OutputManager
from veris.io.storage import Collector, write_snapshot
from veris.physical_constants import PhysicalConstants


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional histories, final snapshots and calendar/sampling controls."""
    parser.add_argument("--netcdf", type=Path, help="Optional sampled netCDF history")
    parser.add_argument(
        "--final-netcdf", type=Path, help="Optional final physical-field snapshot"
    )
    parser.add_argument("--io-variables", default=",".join(Stream().variables))
    parser.add_argument(
        "--sample-seconds",
        type=float,
        help="Sampling interval (default: model timestep)",
    )
    parser.add_argument(
        "--average",
        default="instantaneous",
        help="Comma-separated instantaneous,daily,monthly,annual streams",
    )
    parser.add_argument(
        "--calendar", choices=("gregorian", "360_day", "noleap"), default="gregorian"
    )
    parser.add_argument(
        "--start-date", default="2000-01-01", help="Model origin YYYY-MM-DD"
    )


def make_output(
    args: argparse.Namespace, step_seconds: float, *, collector: Collector | None = None
) -> OutputManager:
    """Create a lazy manager and reject schedules unreachable by the driver."""
    if (
        args.netcdf is not None
        and args.final_netcdf is not None
        and args.netcdf.resolve() == args.final_netcdf.resolve()
    ):
        raise ValueError("history and final snapshot require distinct paths")
    if any(
        path is not None and path.resolve() == args.output.resolve()
        for path in (args.netcdf, args.final_netcdf)
    ):
        raise ValueError("netCDF and NPZ outputs require distinct paths")
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
    return OutputManager(args.netcdf or ".", settings, collector=collector)


def save_final(
    args: argparse.Namespace,
    state: State,
    elapsed: timedelta,
    settings: OutputSettings,
    conf: Configuration,
    phys: PhysicalConstants,
    *,
    collector: Collector | None = None,
) -> None:
    """Write the concrete final State independently of whether history was enabled."""
    if args.final_netcdf is not None:
        write_snapshot(
            args.final_netcdf,
            state,
            elapsed=elapsed,
            start=settings.start,
            calendar=settings.calendar,
            variables=tuple(args.io_variables.split(",")),
            include_halos=False,
            conf=conf,
            phys=phys,
            collector=collector,
        )
