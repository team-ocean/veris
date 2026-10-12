"""Exact independent event oracles for bounded host output planning."""

# ruff: noqa: DTZ001 -- simulation origins are timezone-naive.

from datetime import datetime, timedelta
from typing import Any

import pytest

from veris.io import FixedDate, OutputSettings, Stream


def settings(*streams: Stream, **kwargs: Any) -> OutputSettings:
    """Build minimal schedules from explicitly selected fields."""
    return OutputSettings(streams=streams, **kwargs)


def mean(period: float, interval: float = 1) -> Stream:
    """Build a fixed-duration mean stream."""
    return Stream(
        "mean", ("Area",), timedelta(seconds=interval), timedelta(seconds=period)
    )


def test_independent_events_preserve_exact_completed_windows() -> None:
    from veris.io.schedule import iter_segments

    output = settings(
        Stream("four", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
        Stream("six", ("Area",), timedelta(seconds=2), timedelta(seconds=6)),
        Stream("instant", ("Area",), timedelta(seconds=5)),
    )
    segments = list(iter_segments(output, 1, 13))
    assert [s.start for s in segments] == [0, 4, 5, 6, 8, 10, 12]
    assert [s.stop for s in segments] == [4, 5, 6, 8, 10, 12, 13]
    assert [s.instantaneous for s in segments] == [(), (2,), (), (), (2,), (), ()]
    assert segments[0].completed == ((0, (0, 4_000_000)),)
    assert segments[5].completed == (
        (0, (8_000_000, 12_000_000)),
        (1, (6_000_000, 12_000_000)),
    )
    assert segments[-1].completed == ()


def test_nonaligned_boundaries_retain_microseconds() -> None:
    from veris.io.schedule import iter_segments

    segments = list(iter_segments(settings(mean(2.5)), 1, 8))
    assert [s.stop for s in segments] == [3, 5, 8]
    assert [s.completed for s in segments] == [
        ((0, (0, 2_500_000)),),
        ((0, (2_500_000, 5_000_000)),),
        ((0, (5_000_000, 7_500_000)),),
    ]


def test_cap_splits_without_completing_unfinished_window() -> None:
    from veris.io.schedule import iter_segments

    segments = list(iter_segments(settings(mean(5)), 1, 7, max_steps=2))
    assert [s.stop for s in segments] == [2, 4, 5, 7]
    assert [s.completed for s in segments] == [(), (), ((0, (0, 5_000_000)),), ()]


@pytest.mark.parametrize("initial", [False, True])
def test_empty_substep_windows_are_skipped(initial: bool) -> None:
    from veris.io.schedule import iter_segments

    segments = list(iter_segments(settings(mean(0.1), sample_initial=initial), 1, 3))
    assert [s.stop for s in segments] == ([1, 2, 3] if initial else [2, 3])
    windows = [entry for segment in segments for entry in segment.completed]
    assert windows == [
        (0, (second * 1_000_000, second * 1_000_000 + 100_000))
        for second in range(0 if initial else 1, 3)
    ]


def test_sparse_sampling_skips_windows_without_samples() -> None:
    from veris.io.schedule import iter_segments

    segments = list(iter_segments(settings(mean(2, interval=6)), 1, 9))
    assert [s.stop for s in segments] == [2, 8, 9]
    assert [s.completed for s in segments] == [
        ((0, (0, 2_000_000)),),
        ((0, (6_000_000, 8_000_000)),),
        (),
    ]


@pytest.mark.parametrize(
    "calendar,start,days",
    [
        ("gregorian", datetime(2000, 2, 1), 29),
        ("gregorian", datetime(1900, 2, 1), 28),
        ("noleap", FixedDate(2000, 2, 1), 28),
        ("360_day", FixedDate(2000, 2, 1), 30),
    ],
)
def test_calendar_months(calendar: str, start: Any, days: int) -> None:
    from veris.io.schedule import iter_segments

    output = settings(
        Stream("month", ("Area",), timedelta(days=1), "monthly"),
        calendar=calendar,
        start=start,
    )
    segments = list(iter_segments(output, 86400, days + 1))
    assert [s.stop for s in segments] == [days, days + 1]
    assert segments[0].completed == ((0, (0, days * 86400_000000)),)
    assert segments[1].completed == ()


def test_partial_initial_window_is_flushed_but_identifiable_for_discard() -> None:
    from veris.io.schedule import iter_segments

    output = settings(
        Stream("day", ("Area",), timedelta(hours=6), "daily"),
        start=datetime(2000, 1, 1, 12),
    )
    segments = list(iter_segments(output, 21600, 7))
    assert [s.stop for s in segments] == [2, 6, 7]
    assert segments[0].completed == ((0, (-43200_000000, 43200_000000)),)
    assert segments[1].completed == ((0, (43200_000000, 129600_000000)),)


@pytest.mark.parametrize(
    "step_seconds", [0, -1, float("nan"), float("inf"), True, 0.0000001, 0.0000015]
)
def test_invalid_timestep_even_with_zero_steps(step_seconds: Any) -> None:
    from veris.io.schedule import iter_segments

    with pytest.raises((TypeError, ValueError), match="step|microsecond"):
        list(iter_segments(settings(mean(2)), step_seconds, 0))


@pytest.mark.parametrize(
    "steps,max_steps",
    [(-1, None), (True, None), (1.5, None), (0, 0), (0, True), (0, 1.5)],
)
def test_invalid_counts_even_with_no_output(steps: Any, max_steps: Any) -> None:
    from veris.io.schedule import iter_segments

    with pytest.raises((TypeError, ValueError), match="steps"):
        list(iter_segments(settings(mean(2)), 1, steps, max_steps=max_steps))


def test_incompatible_sampling_rejected_before_iterator_consumption() -> None:
    from veris.io.schedule import iter_segments

    for steps in (0, 5):
        with pytest.raises(ValueError, match="sampling"):
            iter_segments(settings(mean(2, interval=1.5)), 1, steps)


def test_calendar_range_rejected_before_first_segment() -> None:
    from veris.io.schedule import iter_segments

    output = settings(
        Stream("instant", ("Area",), timedelta(days=1)), start=datetime(9999, 12, 31)
    )
    with pytest.raises((OverflowError, ValueError)):
        next(iter(iter_segments(output, 86400, 2)))


def test_zero_steps_and_exact_microsecond_instants() -> None:
    from veris.io.schedule import iter_segments

    output = settings(Stream("instant", ("Area",), timedelta(microseconds=3)))
    assert list(iter_segments(output, 0.000001, 0)) == []
    assert [s.stop for s in iter_segments(output, 0.000001, 7)] == [3, 6, 7]


def test_planner_is_incremental_for_a_long_trajectory() -> None:
    """Planning the first event does not retain a full model-step schedule."""
    from collections.abc import Iterator

    from veris.io.schedule import iter_segments

    segments = iter_segments(settings(mean(4)), 1, 10**10)
    assert isinstance(segments, Iterator)
    first = next(segments)
    second = next(segments)
    assert (first.start, first.stop, second.start, second.stop) == (0, 4, 4, 8)
