"""Host calendar arithmetic for output scheduling, independent of model State.

Gregorian dates use Python's proleptic Gregorian datetime range. Fixed calendars
use integer day and microsecond arithmetic, including dates such as February 30.
Windows contain their left boundary and exclude their right boundary. No model
arrays or numerical kernels participate in this module.
"""

# ruff: noqa: DTZ001 -- model calendar dates are explicitly timezone-naive.

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeAlias

Period: TypeAlias = str | timedelta
_DAY_US = 86400_000000
_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_ALIASES = {
    "standard": "proleptic_gregorian",
    "gregorian": "proleptic_gregorian",
    "proleptic_gregorian": "proleptic_gregorian",
    "360_day": "360_day",
    "noleap": "noleap",
    "365_day": "noleap",
}


def duration_us(duration: timedelta) -> int:
    """Return an exact signed microsecond count without floating-point rounding."""
    if not isinstance(duration, timedelta):
        raise TypeError("duration must be a timedelta")
    return (
        duration.days * 86400 + duration.seconds
    ) * 1_000_000 + duration.microseconds


@dataclass(frozen=True)
class FixedDate:
    """A timezone-free date for a fixed calendar, validated further by Calendar.

    Years start at one. Basic component ranges are checked here; month lengths
    depend on the selected calendar and are checked when constructing Calendar.
    """

    year: int
    month: int
    day: int
    hour: int = 0
    minute: int = 0
    second: int = 0
    microsecond: int = 0

    def __post_init__(self) -> None:
        """Reject noninteger components and impossible basic date/time ranges."""
        limits = {
            "year": (1, None),
            "month": (1, 12),
            "day": (1, 31),
            "hour": (0, 23),
            "minute": (0, 59),
            "second": (0, 59),
            "microsecond": (0, 999999),
        }
        for field, (lower, upper) in limits.items():
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field} must be an integer")
            if value < lower or (upper is not None and value > upper):
                raise ValueError(f"{field} is outside its valid range")


@dataclass(frozen=True, init=False)
class Calendar:
    """A simulation origin and calendar providing exact averaging boundaries.

    Gregorian aliases require a timezone-naive datetime; fixed calendars require
    FixedDate. Daily, monthly and annual windows align to calendar boundaries.
    Positive timedelta windows align to the simulation origin instead.
    """

    start: datetime | FixedDate
    name: str

    def __init__(
        self, start: datetime | FixedDate, calendar: str = "gregorian"
    ) -> None:
        """Validate the date and normalize calendar aliases to CF names."""
        if calendar not in _ALIASES:
            raise ValueError(f"unsupported calendar: {calendar!r}")
        name = _ALIASES[calendar]
        if name == "proleptic_gregorian":
            if not isinstance(start, datetime):
                raise TypeError("Gregorian start must be a datetime")
            if start.tzinfo is not None:
                raise ValueError("model dates must be timezone-naive")
        else:
            if not isinstance(start, FixedDate):
                raise TypeError("fixed-calendar start must be a FixedDate")
            lengths = (30,) * 12 if name == "360_day" else _MONTH_LENGTHS
            if start.day > lengths[start.month - 1]:
                raise ValueError(f"day is invalid for {name} month {start.month}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "name", name)

    @property
    def units(self) -> str:
        """Return CF seconds-since units preserving a subsecond origin."""
        date = self.start
        origin = (
            f"{date.year:04d}-{date.month:02d}-{date.day:02d} "
            f"{date.hour:02d}:{date.minute:02d}:{date.second:02d}"
        )
        if date.microsecond:
            origin += f".{date.microsecond:06d}"
        return f"seconds since {origin}"

    def _fixed_us(self, date: FixedDate) -> int:
        lengths = (30,) * 12 if self.name == "360_day" else _MONTH_LENGTHS
        days = (date.year - 1) * sum(lengths) + sum(lengths[: date.month - 1])
        days += date.day - 1
        seconds = date.hour * 3600 + date.minute * 60 + date.second
        return days * _DAY_US + seconds * 1_000_000 + date.microsecond

    def _fixed_date(self, offset: int) -> FixedDate:
        if offset < 0:
            raise ValueError("fixed-calendar year must be positive")
        days, remainder = divmod(offset, _DAY_US)
        lengths = (30,) * 12 if self.name == "360_day" else _MONTH_LENGTHS
        year, day_of_year = divmod(days, sum(lengths))
        month = 1
        for length in lengths:
            if day_of_year < length:
                break
            day_of_year -= length
            month += 1
        seconds, microsecond = divmod(remainder, 1_000_000)
        hour, seconds = divmod(seconds, 3600)
        minute, second = divmod(seconds, 60)
        return FixedDate(
            year + 1, month, day_of_year + 1, hour, minute, second, microsecond
        )

    def date_at(self, elapsed: timedelta) -> datetime | FixedDate:
        """Return the model date at a signed elapsed time from the origin."""
        offset = duration_us(elapsed)
        if isinstance(self.start, datetime):
            return self.start + elapsed
        return self._fixed_date(self._fixed_us(self.start) + offset)

    def window(self, elapsed: timedelta, period: Period) -> tuple[int, int]:
        """Return containing window bounds as microseconds relative to start.

        Calendar-aligned windows may start before the simulation origin. Samples
        exactly at a right boundary belong to the following window.
        """
        offset = duration_us(elapsed)
        if isinstance(period, timedelta):
            width = duration_us(period)
            if width <= 0:
                raise ValueError("fixed period must be positive")
            lower = (offset // width) * width
            return lower, lower + width
        if period not in ("daily", "monthly", "annual"):
            raise ValueError(f"unsupported averaging period: {period!r}")
        date = self.date_at(elapsed)
        if isinstance(date, datetime):
            assert isinstance(self.start, datetime)
            lower_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            if period == "daily":
                upper_date = lower_date + timedelta(days=1)
            elif period == "monthly":
                lower_date = lower_date.replace(day=1)
                year = date.year + (date.month == 12)
                upper_date = datetime(year, date.month % 12 + 1, 1)
            else:
                lower_date = datetime(date.year, 1, 1)
                upper_date = datetime(date.year + 1, 1, 1)
            return (
                duration_us(lower_date - self.start),
                duration_us(upper_date - self.start),
            )
        assert isinstance(self.start, FixedDate)
        if period == "daily":
            lower = self._fixed_us(FixedDate(date.year, date.month, date.day))
            upper = lower + _DAY_US
        elif period == "monthly":
            lower = self._fixed_us(FixedDate(date.year, date.month, 1))
            upper = self._fixed_us(
                FixedDate(date.year + (date.month == 12), date.month % 12 + 1, 1)
            )
        else:
            lower = self._fixed_us(FixedDate(date.year, 1, 1))
            upper = self._fixed_us(FixedDate(date.year + 1, 1, 1))
        origin = self._fixed_us(self.start)
        return lower - origin, upper - origin
