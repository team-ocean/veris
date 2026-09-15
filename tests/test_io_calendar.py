"""Calendar dates and averaging boundaries have independent scalar expectations."""

# ruff: noqa: DTZ001 -- tests exercise timezone-naive model calendar dates.

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from veris.io.calendar import Calendar, FixedDate, duration_us


@pytest.mark.parametrize("alias", ["gregorian", "standard", "proleptic_gregorian"])
def test_gregorian_aliases_and_exact_time_units(alias: str) -> None:
    calendar = Calendar(datetime(2000, 2, 28, 12, 30, 15, 123456), alias)
    assert calendar.name == "proleptic_gregorian"
    assert calendar.units == "seconds since 2000-02-28 12:30:15.123456"
    assert calendar.date_at(timedelta(days=1)) == datetime(
        2000, 2, 29, 12, 30, 15, 123456
    )


@pytest.mark.parametrize("year, days", [(1900, 28), (2000, 29), (2100, 28)])
def test_gregorian_century_month_lengths(year: int, days: int) -> None:
    calendar = Calendar(datetime(year, 2, 1))
    assert calendar.window(timedelta(), "monthly") == (0, days * 86400_000000)
    assert calendar.date_at(timedelta(days=days)) == datetime(year, 3, 1)


def test_midperiod_windows_and_half_open_boundary_ownership() -> None:
    calendar = Calendar(datetime(2024, 2, 15, 12))
    day = 86400_000000
    assert calendar.window(timedelta(), "daily") == (-day // 2, day // 2)
    assert calendar.window(timedelta(hours=12), "daily") == (day // 2, 3 * day // 2)
    assert calendar.window(timedelta(), "monthly") == (-29 * day // 2, 29 * day // 2)
    assert calendar.window(timedelta(), "annual") == (-91 * day // 2, 641 * day // 2)


def test_fixed_periods_anchor_to_start_and_preserve_microseconds() -> None:
    calendar = Calendar(datetime(2000, 1, 1, 12))
    interval = timedelta(microseconds=7)
    assert calendar.window(timedelta(microseconds=6), interval) == (0, 7)
    assert calendar.window(timedelta(microseconds=7), interval) == (7, 14)
    assert calendar.window(timedelta(microseconds=-1), interval) == (-7, 0)
    assert (
        duration_us(timedelta(days=999999999, microseconds=1)) == 86399999913600000001
    )
    assert duration_us(timedelta(microseconds=-1)) == -1


@pytest.mark.parametrize("alias", ["noleap", "365_day"])
def test_noleap_alias_and_year_transition(alias: str) -> None:
    calendar = Calendar(FixedDate(2000, 2, 28, 23, 59, 59, 999999), alias)
    assert calendar.name == "noleap"
    assert calendar.date_at(timedelta(microseconds=1)) == FixedDate(2000, 3, 1)
    assert calendar.window(timedelta(), "monthly") == (
        -28 * 86400_000000 + 1,
        1,
    )
    new_year = Calendar(FixedDate(2000, 12, 31), alias)
    assert new_year.date_at(timedelta(days=1)) == FixedDate(2001, 1, 1)
    assert new_year.window(timedelta(), "annual") == (-364 * 86400_000000, 86400_000000)


def test_360_day_supports_february_30_and_arithmetic_before_start() -> None:
    calendar = Calendar(FixedDate(2000, 2, 30), "360_day")
    assert calendar.units == "seconds since 2000-02-30 00:00:00"
    assert calendar.date_at(timedelta(days=1)) == FixedDate(2000, 3, 1)
    assert calendar.date_at(timedelta(days=-30)) == FixedDate(2000, 1, 30)
    assert calendar.window(timedelta(), "monthly") == (-29 * 86400_000000, 86400_000000)
    assert calendar.window(timedelta(days=1), "monthly") == (
        86400_000000,
        31 * 86400_000000,
    )
    assert Calendar(FixedDate(2000, 12, 30), "360_day").date_at(
        timedelta(days=1)
    ) == FixedDate(2001, 1, 1)


@pytest.mark.parametrize(
    "period", [timedelta(), timedelta(seconds=-1), "weekly", "instantaneous"]
)
def test_invalid_averaging_periods_are_rejected(period: str | timedelta) -> None:
    with pytest.raises(ValueError, match="period|positive"):
        Calendar(datetime(2000, 1, 1)).window(timedelta(), period)


@pytest.mark.parametrize(
    "date, name",
    [
        (FixedDate(2000, 2, 29), "noleap"),
        (FixedDate(2000, 1, 31), "360_day"),
    ],
)
def test_calendar_specific_dates_are_validated(date: FixedDate, name: str) -> None:
    with pytest.raises(ValueError, match="day"):
        Calendar(date, name)


@pytest.mark.parametrize(
    "field,value",
    [
        ("year", 0),
        ("month", 13),
        ("day", 0),
        ("hour", 24),
        ("minute", 60),
        ("second", 60),
        ("microsecond", 1000000),
        ("year", 2000.5),
    ],
)
def test_fixed_date_component_validation(field: str, value: float) -> None:
    fields: dict[str, Any] = {"year": 2000, "month": 1, "day": 1, field: value}
    with pytest.raises((TypeError, ValueError), match=field):
        FixedDate(**fields)


def test_fixed_date_is_frozen() -> None:
    date = FixedDate(2000, 1, 1)
    with pytest.raises(FrozenInstanceError):
        cast(Any, date).year = 2001  # Deliberately exercise frozen assignment.


def test_rejects_timezone_and_mismatched_date_types() -> None:
    with pytest.raises(ValueError, match="timezone|naive"):
        Calendar(datetime(2000, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(TypeError, match="FixedDate"):
        Calendar(datetime(2000, 1, 1), "360_day")
    with pytest.raises(TypeError, match="datetime"):
        Calendar(FixedDate(2000, 1, 1), "gregorian")
    with pytest.raises(ValueError, match="calendar"):
        Calendar(datetime(2000, 1, 1), "julian")


def test_time_inputs_require_timedelta_and_fixed_years_stay_positive() -> None:
    with pytest.raises(TypeError, match="timedelta"):
        duration_us(cast(Any, 1.0))  # Deliberately invalid runtime input.
    with pytest.raises(ValueError, match="year"):
        Calendar(FixedDate(1, 1, 1), "360_day").date_at(timedelta(microseconds=-1))
