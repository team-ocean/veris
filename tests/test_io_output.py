"""Independent sample-mean oracles, schedule validation and AD no-effect checks."""

# ruff: noqa: DTZ001 -- model calendar dates are explicitly timezone-naive.

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from veris.io.calendar import FixedDate


def fields(value: float) -> dict[str, NDArray[Any]]:
    return {"Area": np.full((6, 7), value), "hIceMean": np.full((6, 7), value * 2)}


def test_daily_boundary_sample_starts_new_window_and_streams_select_fields(
    tmp_path: Path,
) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    path = tmp_path / "history.nc"
    settings = OutputSettings(
        streams=(
            Stream("instant", ("Area",), timedelta(hours=12)),
            Stream("day", ("hIceMean",), timedelta(hours=12), "daily"),
        )
    )
    with OutputManager(path, settings) as output:
        for hour, value in [(0, 2), (12, 6), (24, 10), (36, 14)]:
            output.sample(fields(value), timedelta(hours=hour))
        output.close(timedelta(hours=48))
    first = read_record(path, stream="day", index=0)
    np.testing.assert_array_equal(first.fields["hIceMean"], np.full((2, 3), 8.0))
    assert first.bounds == (0, 86400) and first.count == 2 and not first.partial
    second = read_record(path, stream="day", index=1)
    np.testing.assert_array_equal(second.fields["hIceMean"], np.full((2, 3), 24.0))
    assert second.bounds == (86400, 172800) and not second.partial
    with h5netcdf.File(path) as file:
        assert len(file.groups["instant"].dimensions["time"]) == 4
        assert "hIceMean" not in file.groups["instant"].variables
        np.testing.assert_array_equal(
            file.groups["instant"].variables["time"][:], [0, 43200, 86400, 129600]
        )


@pytest.mark.parametrize(
    "calendar,start,days",
    [
        ("gregorian", datetime(2000, 2, 1), 29),
        ("gregorian", datetime(1900, 2, 1), 28),
        ("360_day", (2000, 2, 1), 30),
        ("noleap", (2000, 2, 1), 28),
    ],
)
def test_monthly_mean_uses_calendar_window(
    tmp_path: Path,
    calendar: str,
    start: datetime | tuple[int, int, int] | FixedDate,
    days: int,
) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    if isinstance(start, tuple):
        start = FixedDate(*start)
    settings = OutputSettings(
        start=start,
        calendar=calendar,
        streams=(Stream("month", ("Area",), timedelta(days=1), "monthly"),),
    )
    with OutputManager(tmp_path / "month.nc", settings) as output:
        for day in range(days):
            output.sample(fields(day), timedelta(days=day))
        output.close(timedelta(days=days))
    record = read_record(tmp_path / "month.nc", stream="month")
    np.testing.assert_array_equal(
        record.fields["Area"], np.full((2, 3), (days - 1) / 2)
    )
    assert (
        record.bounds == (0, days * 86400)
        and record.count == days
        and not record.partial
    )


def test_initial_and_final_incomplete_windows_create_no_file(tmp_path: Path) -> None:
    from veris.io import OutputManager, OutputSettings, Stream

    path = tmp_path / "partial.nc"
    settings = OutputSettings(
        start=datetime(2001, 1, 1, 12),
        streams=(Stream("day", ("Area",), timedelta(hours=6), "daily"),),
    )
    with OutputManager(path, settings) as output:
        output.sample(fields(2), timedelta(0))
        output.sample(fields(6), timedelta(hours=6))
        output.sample(fields(20), timedelta(hours=12))
        output.close(timedelta(hours=15))
    assert not path.exists()
    assert output.buffer_nbytes == 0


def test_fixed_period_non_sampling_boundary_and_empty_windows(tmp_path: Path) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    settings = OutputSettings(
        streams=(Stream("fixed", ("Area",), timedelta(hours=6), timedelta(hours=2)),)
    )
    with OutputManager(tmp_path / "fixed.nc", settings) as output:
        output.sample(fields(3), timedelta(0))
        output.sample({}, timedelta(hours=2))  # flush without touching arrays
        output.sample(fields(9), timedelta(hours=6))
        output.close(timedelta(hours=8))
    with h5netcdf.File(tmp_path / "fixed.nc") as file:
        assert len(file.groups["fixed"].dimensions["time"]) == 2
    assert read_record(tmp_path / "fixed.nc", stream="fixed", index=1).bounds == (
        21600,
        28800,
    )


def test_failed_schedule_validation_does_not_advance_clock(tmp_path: Path) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    settings = OutputSettings(
        streams=(Stream("instant", ("Area",), timedelta(seconds=2)),)
    )
    with OutputManager(tmp_path / "schedule.nc", settings) as output:
        output.sample(fields(1), timedelta(0))
        with pytest.raises(ValueError, match="missed"):
            output.sample(fields(99), timedelta(seconds=3))
        output.sample(fields(2), timedelta(seconds=2))
        with pytest.raises(ValueError, match="increas"):
            output.sample(fields(99), timedelta(seconds=2))
        with pytest.raises(ValueError):
            output.sample({"unknown": np.zeros((6, 7))}, timedelta(seconds=4))
        output.sample(fields(3), timedelta(seconds=4))
    assert read_record(tmp_path / "schedule.nc", stream="instant").time == 4


def test_no_initial_sample_and_discard_partial_do_not_create_empty_file(
    tmp_path: Path,
) -> None:
    from veris.io import OutputManager, OutputSettings, Stream

    settings = OutputSettings(
        sample_initial=False,
        streams=(Stream("day", ("Area",), timedelta(hours=1), "daily"),),
    )
    with OutputManager(tmp_path / "none.nc", settings) as output:
        output.sample({}, timedelta(0))
        output.sample(fields(2), timedelta(hours=1))
    assert not (tmp_path / "none.nc").exists()


def test_mean_buffer_memory_does_not_grow_with_samples(tmp_path: Path) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    settings = OutputSettings(
        streams=(
            Stream("annual", ("Area",), timedelta(seconds=1), timedelta(seconds=1001)),
        )
    )
    with OutputManager(tmp_path / "bounded.nc", settings) as output:
        for second in range(1001):
            output.sample(fields(second), timedelta(seconds=second))
            assert output.buffer_nbytes == 2 * 3 * 8
        output.close(timedelta(seconds=1001))
    record = read_record(tmp_path / "bounded.nc", stream="annual")
    np.testing.assert_array_equal(record.fields["Area"], np.full((2, 3), 500.0))
    assert record.count == 1001


@pytest.mark.parametrize(
    "disabled,transform",
    [(False, "grad"), (False, "jvp"), (False, "jit"), (True, "grad")],
)
def test_transforms_and_explicit_ad_mode_do_not_sample_even_constant_fields(
    tmp_path: Path, disabled: bool, transform: str
) -> None:
    """Each trace guard is exercised; disabled output bypasses all trace guards."""
    from veris.io import OutputManager, OutputSettings, Stream

    path = tmp_path / "ad.nc"
    settings = OutputSettings(
        enabled=not disabled,
        streams=(Stream("day", ("Area",), timedelta(seconds=1), timedelta(seconds=1)),),
    )
    with OutputManager(path, settings) as output:

        def objective(x: float | jax.Array) -> float | jax.Array:
            output.sample(fields(7), timedelta(0))
            output.sample({"Area": jnp.ones((6, 7)) * x}, timedelta(seconds=1))
            output.close()
            return x * x

        if transform == "grad":
            assert jax.grad(objective)(2.0) == 4
        elif transform == "jvp":
            assert jax.jvp(objective, (2.0,), (1.0,))[1] == 4
        else:
            assert jax.jit(objective)(2.0) == 4
        assert output.buffer_nbytes == 0 and not path.exists()
        if disabled:
            output.sample(cast(Any, object()), cast(Any, "invalid time"))
        else:
            output.sample(
                fields(2), timedelta(0)
            )  # tracing did not change the schedule
            output.close(timedelta(seconds=1))
    assert path.exists() == (not disabled)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampling_interval": timedelta(0)},
        {"sampling_interval": timedelta(seconds=-1)},
        {"period": "weekly"},
        {"period": timedelta(0)},
        {"variables": ()},
        {"variables": ("unknown",)},
        {"variables": ("Area", "Area")},
        {"name": "../bad"},
    ],
)
def test_invalid_stream_configuration(kwargs: dict[str, Any]) -> None:
    from veris.io import Stream

    options: dict[str, Any] = {"name": "ice", "variables": ("Area",)} | kwargs
    with pytest.raises((ValueError, TypeError)):
        Stream(**options)


@pytest.mark.parametrize(
    "calendar,days", [("gregorian", 366), ("360_day", 360), ("noleap", 365)]
)
def test_annual_sample_mean_and_year_rollover(
    tmp_path: Path, calendar: str, days: int
) -> None:
    from veris.io import FixedDate, OutputManager, OutputSettings, Stream, read_record

    start = datetime(2000, 1, 1) if calendar == "gregorian" else FixedDate(2000, 1, 1)
    settings = OutputSettings(
        start=start,
        calendar=calendar,
        streams=(Stream("year", ("Area",), timedelta(days=1), "annual"),),
    )
    with OutputManager(tmp_path / "annual.nc", settings) as output:
        for day in range(days + 1):
            output.sample(fields(day), timedelta(days=day))
    record = read_record(tmp_path / "annual.nc", stream="year", index=0)
    assert record.count == days and not record.partial
    assert record.bounds == (0, days * 86400)
    np.testing.assert_array_equal(
        record.fields["Area"], np.full((2, 3), (days - 1) / 2)
    )
    with h5netcdf.File(tmp_path / "annual.nc") as file:
        assert len(file.groups["year"].dimensions["time"]) == 1


def test_independent_subsecond_schedules(tmp_path: Path) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    settings = OutputSettings(
        streams=(
            Stream("fast", ("Area",), timedelta(microseconds=100000)),
            Stream(
                "slow",
                ("hIceMean",),
                timedelta(microseconds=200000),
                timedelta(microseconds=500000),
            ),
        )
    )
    with OutputManager(tmp_path / "micro.nc", settings) as output:
        for step in range(5):
            output.sample(fields(step), timedelta(microseconds=step * 100000))
        output.close(timedelta(microseconds=500000))
    record = read_record(tmp_path / "micro.nc", stream="slow")
    assert record.count == 3
    np.testing.assert_array_equal(record.fields["hIceMean"], np.full((2, 3), 4.0))
    with h5netcdf.File(tmp_path / "micro.nc") as file:
        np.testing.assert_allclose(
            file.groups["fast"].variables["time"][:], [0, 0.1, 0.2, 0.3, 0.4]
        )


def test_only_complete_window_between_partial_windows_is_written(
    tmp_path: Path,
) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    path = tmp_path / "complete.nc"
    settings = OutputSettings(
        start=datetime(2000, 1, 1, 12),
        streams=(Stream("day", ("Area",), timedelta(hours=12), "daily"),),
    )
    with OutputManager(path, settings) as output:
        for hour, value in [(0, 100), (12, 3), (24, 9), (36, 200)]:
            output.sample(fields(value), timedelta(hours=hour))
        output.close(timedelta(hours=42))
    record = read_record(path, stream="day")
    assert not record.partial and record.bounds == (43200, 129600)
    assert record.count == 2
    np.testing.assert_array_equal(record.fields["Area"], np.full((2, 3), 6.0))
    with h5netcdf.File(path) as file:
        assert len(file.groups["day"].dimensions["time"]) == 1


@pytest.mark.parametrize("removed_option", ["include_halos", "write_partial"])
def test_removed_output_options_are_not_accepted(removed_option: str) -> None:
    from veris.io import OUTPUT_SETTINGS, OutputSettings

    assert removed_option not in OUTPUT_SETTINGS
    options: dict[str, Any] = {removed_option: True}
    with pytest.raises(TypeError):
        OutputSettings(**options)


def test_rejected_dtype_cannot_contaminate_an_earlier_mean_stream(
    tmp_path: Path,
) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    settings = OutputSettings(
        streams=(
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=2)),
            Stream("instant", ("Area",), timedelta(seconds=1)),
        )
    )
    with OutputManager(tmp_path / "retry.nc", settings) as output:
        output.sample({"Area": np.ones((6, 7), dtype="int32")}, timedelta(0))
        with pytest.raises(ValueError, match="dtype"):
            output.sample({"Area": np.full((6, 7), 1.5)}, timedelta(seconds=1))
        output.sample({"Area": np.full((6, 7), 2, dtype="int32")}, timedelta(seconds=1))
        output.close(timedelta(seconds=2))
    record = read_record(tmp_path / "retry.nc", stream="mean")
    np.testing.assert_array_equal(record.fields["Area"], np.full((2, 3), 1.5))
    assert record.count == 2


def test_float32_samples_accumulate_in_float64(tmp_path: Path) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    settings = OutputSettings(
        streams=(Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=3)),)
    )
    with OutputManager(tmp_path / "precision.nc", settings) as output:
        for step, value in enumerate([2**24, 1, -(2**24)]):
            output.sample(
                {"Area": np.full((6, 7), value, dtype="float32")},
                timedelta(seconds=step),
            )
        output.close(timedelta(seconds=3))
    record = read_record(tmp_path / "precision.nc", stream="mean")
    assert record.fields["Area"].dtype == np.float64
    np.testing.assert_array_equal(record.fields["Area"], np.full((2, 3), 1 / 3))


def test_collector_returns_physical_cells_without_further_trimming(
    tmp_path: Path,
) -> None:
    from veris.io import OutputManager, OutputSettings, Stream, read_record

    physical = np.arange(30.0).reshape(5, 6)
    source = {"Area": np.pad(physical, 2, constant_values=-999)}
    calls = []

    def collect(state: Any, names: tuple[str, ...]) -> dict[str, NDArray[Any]]:
        calls.append((state, names))
        return {"Area": physical}

    path = tmp_path / "collected.nc"
    settings = OutputSettings(streams=(Stream("instant", ("Area",)),))
    with OutputManager(path, settings, collector=collect) as output:
        output.sample(source, timedelta(0))
    assert len(calls) == 1 and calls[0][0] is source
    assert calls[0][1] == ("Area",)
    np.testing.assert_array_equal(
        read_record(path, stream="instant").fields["Area"], physical
    )
