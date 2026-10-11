"""Plan exact calendar output events without retaining a timestep schedule.

Segments carry model-step endpoints and exact microsecond mean bounds. A mean
window's execution endpoint rounds upward to a model time, while its metadata
retains the original calendar boundary. Empty windows are skipped by jumping to
the next scheduled sample. Partial initial windows remain events so callers can
reset their sums, but their negative left bound identifies them for discarding.
This host-only layer performs no array operations or model transitions.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from numbers import Integral, Real

from veris.io.calendar import Calendar, duration_us
from veris.io.configuration import OutputSettings


@dataclass(frozen=True)
class Segment:
    """A scan interval and output events completed at its right endpoint."""

    start: int
    stop: int
    completed: tuple[tuple[int, tuple[int, int]], ...]
    instantaneous: tuple[int, ...]


def timestep_us(step_seconds: float) -> int:
    """Validate a positive timestep exactly expressible in microseconds."""
    if isinstance(step_seconds, bool) or not isinstance(step_seconds, Real):
        raise TypeError("step_seconds must be a real number")
    microseconds = Decimal(str(step_seconds)) * 1_000_000
    if not microseconds.is_finite() or microseconds <= 0:
        raise ValueError("step_seconds must be positive and finite")
    if microseconds != microseconds.to_integral_value():
        raise ValueError("step_seconds must be an integer number of microseconds")
    return int(microseconds)


def _count(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def iter_segments(
    settings: OutputSettings,
    step_seconds: float,
    steps: int,
    max_steps: int | None = None,
    *,
    start_step: int = 0,
    pending_windows: tuple[tuple[int, int] | None, ...] | None = None,
) -> Iterator[Segment]:
    """Yield event-aligned chunks after validating the entire run's time range.

    Sampling intervals must be exact multiples of the model timestep. Mean
    periods need not align: a boundary between model times completes at the
    following step, with its exact bounds unchanged. Initial instantaneous
    output is handled separately by the caller. Storage scales with the number
    of streams, independently of trajectory length. Validation also applies to
    zero-step runs and happens when this function is called.
    """
    steps = _count("steps", steps, 0)
    start_step = _count("start_step", start_step, 0)
    if max_steps is not None:
        max_steps = _count("max_steps", max_steps, 1)
    step_us = timestep_us(step_seconds)
    intervals = tuple(duration_us(s.sampling_interval) for s in settings.streams)
    if any(interval % step_us for interval in intervals):
        raise ValueError("sampling intervals must be multiples of step_seconds")
    clock = Calendar(settings.start, settings.calendar)
    start_us = start_step * step_us
    end_step = start_step + steps
    end_us = end_step * step_us
    if pending_windows is not None:
        if not isinstance(pending_windows, tuple) or len(pending_windows) != len(
            settings.streams
        ):
            raise ValueError("pending_windows must match configured streams")
        for stream, window in zip(settings.streams, pending_windows, strict=True):
            if window is None:
                continue
            if (
                stream.period == "instantaneous"
                or not isinstance(window, tuple)
                or len(window) != 2
                or any(
                    isinstance(v, bool) or not isinstance(v, Integral) for v in window
                )
                or window[0] >= window[1]
                or window[1] <= start_us
            ):
                raise ValueError("pending_windows must contain unfinished mean bounds")
            if (
                clock.window(
                    timedelta(microseconds=max(start_us, window[0])), stream.period
                )
                != window
            ):
                raise ValueError("pending_windows bounds do not match calendar windows")
    end = timedelta(microseconds=end_us)
    clock.date_at(end)
    # Preflight the endpoint window too, so a calendar range error cannot occur
    # after an earlier segment has already advanced the requested trajectory.
    for stream in settings.streams:
        if stream.period != "instantaneous":
            clock.window(end, stream.period)

    def window_at(index: int, sample_us: int) -> tuple[int, int] | None:
        """Locate a nonempty candidate, excluding the final right endpoint."""
        if sample_us >= end_us:
            return None
        return clock.window(
            timedelta(microseconds=sample_us), settings.streams[index].period
        )

    windows = {}
    instants = {}
    for index, stream in enumerate(settings.streams):
        interval = intervals[index]
        if stream.period == "instantaneous":
            instants[index] = ((start_us // interval) + 1) * interval // step_us
        else:
            sample = ((start_us + interval - 1) // interval) * interval
            if not settings.sample_initial and sample == 0:
                sample = interval
            retained = None if pending_windows is None else pending_windows[index]
            windows[index] = (
                retained if retained is not None else window_at(index, sample)
            )

    def generate() -> Iterator[Segment]:
        start = start_step
        while start < end_step:
            stop = end_step if max_steps is None else min(end_step, start + max_steps)
            for window in windows.values():
                if window is not None and window[1] <= end_us:
                    stop = min(stop, (window[1] + step_us - 1) // step_us)
            stop = min((stop, *instants.values()))
            completed = tuple(
                (index, window)
                for index, window in windows.items()
                if window is not None and window[1] <= stop * step_us
            )
            instantaneous = tuple(
                index for index, due in instants.items() if due == stop
            )
            yield Segment(start, stop, completed, instantaneous)
            for index, window in completed:
                interval = intervals[index]
                next_sample = ((window[1] + interval - 1) // interval) * interval
                windows[index] = window_at(index, next_sample)
            for index in instantaneous:
                instants[index] += intervals[index] // step_us
            start = stop

    return generate()
