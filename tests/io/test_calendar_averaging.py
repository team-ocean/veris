"""Independent calendar means across whole years and resumed compiled rollouts.

Literal month lengths are the oracle, rather than Calendar or the direct writer.
Nonlinear daily fields expose samples leaking between windows; a two-day sampling
cadence exposes phase errors across odd-length months. Real netCDF records verify
counts, bounds, midpoints, calendar metadata and physical cells without halos.
"""

# ruff: noqa: DTZ001 -- simulation calendar dates are timezone-naive.

from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from veris.execution.rollout_output import output_callbacks, run_timed
from veris.io import (
    Calendar,
    FixedDate,
    OutputManager,
    OutputSettings,
    Stream,
    read_record,
)

_COMMON_MONTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_LEAP_MONTHS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_CASES = (
    pytest.param("gregorian", 1900, _COMMON_MONTHS * 2, id="gregorian-century"),
    pytest.param("gregorian", 2000, _LEAP_MONTHS + _COMMON_MONTHS, id="gregorian-leap"),
    pytest.param("360_day", 2000, (30,) * 24, id="360-day"),
    pytest.param("noleap", 2000, _COMMON_MONTHS * 2, id="noleap"),
)
_DAY_US = 86400_000000
_PHYSICAL = np.arange(6, dtype=np.float64).reshape(2, 3)


def _origin(
    calendar: str, year: int, day: int = 1, hour: int = 0
) -> datetime | FixedDate:
    """Construct only the date type required by the public calendar contract."""
    return (
        datetime(year, 1, day, hour)
        if calendar == "gregorian"
        else FixedDate(year, 1, day, hour)
    )


@pytest.mark.parametrize("calendar,year,lengths", _CASES)
def test_every_month_has_independent_bounds_and_half_open_ownership(
    calendar: str, year: int, lengths: tuple[int, ...]
) -> None:
    """Wrong month tables, leap rules or December rollover change these bounds."""
    clock = Calendar(_origin(calendar, year), calendar)
    elapsed_days = 0
    for index, days in enumerate(lengths):
        left, right = elapsed_days * _DAY_US, (elapsed_days + days) * _DAY_US
        assert clock.window(timedelta(microseconds=left), "monthly") == (left, right)
        assert clock.window(timedelta(microseconds=right - 1), "monthly") == (
            left,
            right,
        )
        last = clock.date_at(timedelta(microseconds=right - 1))
        assert (last.year, last.month, last.day) == (
            year + index // 12,
            index % 12 + 1,
            days,
        )
        following = clock.date_at(timedelta(microseconds=right))
        assert (following.year, following.month, following.day) == (
            year + (index + 1) // 12,
            (index + 1) % 12 + 1,
            1,
        )
        elapsed_days += days


def _field(day: int) -> NDArray[np.float64]:
    """Embed a nonlinear time series in physical cells with sentinel halos."""
    return np.pad(_PHYSICAL + day**2, 2, constant_values=-999)


def _advance(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Evolve the same quadratic series inside the actual compiled reducer."""
    return {
        "hIceMean": state["hIceMean"] + 2 * state["day"] + 1,
        "day": state["day"] + 1,
    }


@pytest.mark.parametrize("calendar,year,lengths", _CASES)
@pytest.mark.parametrize("origin_days", [0.0, 14.5], ids=["aligned", "midmonth"])
@pytest.mark.parametrize("mode", ["direct", "scheduled", "resumed"])
def test_monthly_and_annual_means_follow_independent_calendar_samples(
    tmp_path: Path,
    calendar: str,
    year: int,
    lengths: tuple[int, ...],
    origin_days: float,
    mode: str,
) -> None:
    """Catch wrong durations, boundary leakage, cadence reset and partial writes."""
    settings = OutputSettings(
        start=_origin(calendar, year, 15, 12)
        if origin_days
        else _origin(calendar, year),
        calendar=calendar,
        streams=(
            Stream("monthly", ("hIceMean",), timedelta(days=2), "monthly"),
            Stream("annual", ("hIceMean",), timedelta(days=2), "annual"),
        ),
    )
    steps = sum(lengths) if origin_days == 0 else int(sum(lengths) - origin_days) + 10
    path = tmp_path / "calendar.nc"
    with OutputManager(path, settings) as manager:
        if mode == "direct":
            for day in range(0, steps, 2):
                manager.sample({"hIceMean": _field(day)}, timedelta(days=day))
            manager.close(timedelta(days=steps))
        else:
            observer, select = output_callbacks(
                manager, 86400, continue_runs=mode == "resumed"
            )
            state = {"hIceMean": jnp.asarray(_field(0)), "day": jnp.array(0)}
            first_year = sum(lengths[:12])
            endpoints = (
                (
                    17,
                    30,
                    31,
                    45,
                    59,
                    60,
                    first_year - 1,
                    first_year + 1,
                    steps - 15,
                    steps,
                )
                if mode == "resumed"
                else (steps,)
            )
            start = 0
            for stop in endpoints:
                state, _, _ = run_timed(
                    state, _advance, stop - start, observe=observer, select=select
                )
                start = stop
            assert int(state["day"]) == steps
            np.testing.assert_array_equal(
                state["hIceMean"][2:-2, 2:-2], _PHYSICAL + steps**2
            )

    samples = np.arange(0, steps, 2)
    month_edges = np.concatenate(([0], np.cumsum(lengths))) - origin_days
    year_edges = month_edges[::12]
    for stream, edges in (("monthly", month_edges), ("annual", year_edges)):
        complete = [
            (left, right)
            for left, right in pairwise(edges)
            if left >= 0 and right <= steps
        ]
        with h5netcdf.File(path) as file:
            assert len(file.groups[stream].dimensions["time"]) == len(complete)
        for index, (left, right) in enumerate(complete):
            selected = samples[(samples >= left) & (samples < right)]
            record = read_record(path, stream=stream, index=index)
            assert record.count == len(selected)
            assert record.bounds == (left * 86400, right * 86400)
            assert record.time == (left + right) * 43200
            assert record.calendar == (
                "proleptic_gregorian" if calendar == "gregorian" else calendar
            )
            assert record.mean and not record.partial
            expected = _PHYSICAL + np.mean(selected**2)
            actual = record.fields["hIceMean"]
            assert actual.shape == expected.shape
            max_error = float(np.max(np.abs(actual - expected)))
            # Compiled division may multiply by a reciprocal, differing from
            # the host scalar oracle by one float64 ULP for noninteger means.
            tolerance = float(2 * np.spacing(np.max(expected)))
            assert max_error <= tolerance, (
                f"ERROR {calendar} {mode} {stream} record {index}: "
                f"max error {max_error:g} exceeds {tolerance:g}"
            )
