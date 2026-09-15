"""Immutable host output controls and registry metadata for generated docs."""

# ruff: noqa: DTZ001 -- model dates intentionally have no timezone.

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, NamedTuple

from veris.io.calendar import Calendar, FixedDate, Period, duration_us
from veris.variables import VARIABLES


class OutputOption(NamedTuple):
    """Default, Python type and description of a host-only output option."""

    default: Any
    type: type
    description: str


STREAM_SETTINGS = {
    "name": OutputOption("instantaneous", str, "Unique netCDF group name."),
    "variables": OutputOption(
        ("hIceMean", "hSnowMean", "Area", "uIce", "vIce"),
        tuple,
        "Selected names from VARIABLES.",
    ),
    "sampling_interval": OutputOption(
        timedelta(hours=1),
        timedelta,
        "Positive interval between samples, anchored to simulation start.",
    ),
    "period": OutputOption(
        "instantaneous",
        str,
        "instantaneous, daily, monthly, annual, or a positive timedelta.",
    ),
}


@dataclass(frozen=True)
class Stream:
    """One variable selection with its own sampling and averaging schedule."""

    name: str = STREAM_SETTINGS["name"].default
    variables: tuple[str, ...] = STREAM_SETTINGS["variables"].default
    sampling_interval: timedelta = STREAM_SETTINGS["sampling_interval"].default
    period: Period = STREAM_SETTINGS["period"].default

    def __post_init__(self) -> None:
        """Normalize selection and reject ambiguous or invalid schedules."""
        if not isinstance(self.name, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", self.name
        ):
            raise ValueError("stream name must be a simple netCDF identifier")
        if isinstance(self.variables, str):
            raise TypeError("variables must be a sequence of names")
        object.__setattr__(self, "variables", tuple(self.variables))
        if (
            not self.variables
            or len(set(self.variables)) != len(self.variables)
            or any(name not in VARIABLES for name in self.variables)
        ):
            raise ValueError(
                "variables must be a nonempty unique selection from VARIABLES"
            )
        if duration_us(self.sampling_interval) <= 0:
            raise ValueError("sampling interval must be positive")
        if isinstance(self.period, timedelta):
            if duration_us(self.period) <= 0:
                raise ValueError("averaging period must be positive")
        elif self.period not in ("instantaneous", "daily", "monthly", "annual"):
            raise ValueError("unsupported averaging period")


OUTPUT_SETTINGS = {
    "start": OutputOption(
        datetime(2000, 1, 1),
        datetime,
        "Simulation origin: naive datetime for Gregorian, FixedDate for fixed calendars.",
    ),
    "calendar": OutputOption(
        "gregorian", str, "gregorian (proleptic), 360_day, or noleap (365_day alias)."
    ),
    "streams": OutputOption((Stream(),), tuple, "Independent Stream configurations."),
    "sample_initial": OutputOption(
        True, bool, "Include the initial State at elapsed zero."
    ),
    "enabled": OutputOption(
        True,
        bool,
        "Set False during AD: no sampling, clock changes, array transfers or files.",
    ),
}


@dataclass(frozen=True)
class OutputSettings:
    """Output-only configuration allocated separately from numerical State."""

    start: datetime | FixedDate = OUTPUT_SETTINGS["start"].default
    calendar: str = OUTPUT_SETTINGS["calendar"].default
    streams: tuple[Stream, ...] = OUTPUT_SETTINGS["streams"].default
    sample_initial: bool = OUTPUT_SETTINGS["sample_initial"].default
    enabled: bool = OUTPUT_SETTINGS["enabled"].default

    def __post_init__(self) -> None:
        """Validate calendars, unique streams and strict Boolean controls."""
        Calendar(self.start, self.calendar)
        object.__setattr__(self, "streams", tuple(self.streams))
        if not self.streams or any(not isinstance(s, Stream) for s in self.streams):
            raise ValueError("at least one Stream is required")
        if len({s.name for s in self.streams}) != len(self.streams):
            raise ValueError("stream names must be unique")
        for name in ("sample_initial", "enabled"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
