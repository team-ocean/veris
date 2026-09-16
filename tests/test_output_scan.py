"""Scheduled output agrees with independent direct samples without histories."""

# ruff: noqa: DTZ001 -- model calendars have timezone-naive origins.

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.integration_output import output_callbacks, run_timed
from veris.io import FixedDate, OutputManager, OutputSettings, Stream, read_record


def _fields(iteration: int) -> dict[str, Any]:
    """Use spatially varying fields to detect accidental halo or axis changes."""
    pattern = np.arange(42.0).reshape(6, 7)
    return {"Area": pattern + iteration, "hIceMean": 2 * pattern + 3 * iteration}


def _advance(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
    return {"Area": state["Area"] + 1, "hIceMean": state["hIceMean"] + 3}


def _assert_files_equal(actual: Path, expected: Path) -> None:
    """Compare all records and metadata, including absent partial-only output."""
    assert actual.exists() == expected.exists()
    if not expected.exists():
        return
    with h5netcdf.File(actual) as left, h5netcdf.File(expected) as right:
        assert set(left.groups) == set(right.groups)
        for name, oracle in right.groups.items():
            result = left.groups[name]
            assert dict(result.attrs) == dict(oracle.attrs)
            assert set(result.variables) == set(oracle.variables)
            for variable, reference in oracle.variables.items():
                assert dict(result.variables[variable].attrs) == dict(reference.attrs)
                np.testing.assert_array_equal(
                    result.variables[variable][:], reference[:]
                )


def _compare_direct(
    tmp_path: Path,
    settings: OutputSettings,
    step_seconds: float,
    steps: int,
    cap: int | None = None,
) -> None:
    expected, actual = tmp_path / "direct.nc", tmp_path / "scheduled.nc"
    with OutputManager(expected, settings) as manager:
        for iteration in range(steps + 1):
            manager.sample(
                _fields(iteration), timedelta(seconds=iteration * step_seconds)
            )
    initial = jax.tree.map(jnp.asarray, _fields(0))
    with OutputManager(actual, settings) as manager:
        observe, select = output_callbacks(manager, step_seconds)
        kwargs: dict[str, Any] = {} if cap is None else {"chunk_size": cap}
        final, compiled, elapsed = run_timed(
            initial, _advance, steps, observe=observe, select=select, **kwargs
        )
    for name, expected_field in _fields(steps).items():
        np.testing.assert_array_equal(final[name], expected_field)
    assert compiled >= 0 and elapsed >= 0
    _assert_files_equal(actual, expected)


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("cap", [None, 2])
def test_concurrent_cadences_and_windows_match_direct(
    tmp_path: Path, initial: bool, cap: int | None
) -> None:
    output = OutputSettings(
        sample_initial=initial,
        streams=(
            Stream("four", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
            Stream(
                "six", ("Area", "hIceMean"), timedelta(seconds=2), timedelta(seconds=6)
            ),
            Stream("instant", ("hIceMean",), timedelta(seconds=5)),
        ),
    )
    _compare_direct(tmp_path, output, 1, 13, cap)


@pytest.mark.parametrize(
    "period,interval,steps", [(2.5, 1, 8), (3, 2, 7), (0.1, 1, 4), (2, 6, 9)]
)
def test_nonaligned_sparse_and_substep_windows_match_direct(
    tmp_path: Path, period: float, interval: float, steps: int
) -> None:
    output = OutputSettings(
        streams=(
            Stream(
                "mean",
                ("Area",),
                timedelta(seconds=interval),
                timedelta(seconds=period),
            ),
        )
    )
    _compare_direct(tmp_path, output, 1, steps)


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("steps", [0, 3, 4])
def test_zero_steps_partial_final_and_exact_final_boundary(
    tmp_path: Path, initial: bool, steps: int
) -> None:
    output = OutputSettings(
        sample_initial=initial,
        streams=(
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
        ),
    )
    _compare_direct(tmp_path, output, 1, steps)


@pytest.mark.parametrize(
    "calendar,start,days",
    [
        ("gregorian", datetime(2000, 2, 1), 29),
        ("gregorian", datetime(1900, 2, 1), 28),
        ("noleap", FixedDate(2000, 2, 1), 28),
        ("360_day", FixedDate(2000, 2, 1), 30),
    ],
)
def test_calendar_month_records_match_direct(
    tmp_path: Path, calendar: str, start: Any, days: int
) -> None:
    output = OutputSettings(
        start=start,
        calendar=calendar,
        streams=(Stream("month", ("Area",), timedelta(days=1), "monthly"),),
    )
    _compare_direct(tmp_path, output, 86400, days + 2)


@pytest.mark.parametrize(
    "calendar,start",
    [
        ("gregorian", datetime(2000, 1, 1, 12)),
        ("noleap", FixedDate(2000, 1, 1, 12)),
        ("360_day", FixedDate(2000, 1, 1, 12)),
    ],
)
def test_calendar_partial_first_and_last_windows_match_direct(
    tmp_path: Path, calendar: str, start: Any
) -> None:
    output = OutputSettings(
        start=start,
        calendar=calendar,
        streams=(Stream("day", ("Area",), timedelta(hours=6), "daily"),),
    )
    _compare_direct(tmp_path, output, 21600, 7)


@pytest.mark.parametrize("x64", [False, True])
def test_float32_cancellation_and_physics_precision_are_preserved(
    tmp_path: Path, x64: bool
) -> None:
    trace_modes = []
    output = OutputSettings(
        streams=(Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=3)),)
    )
    path = tmp_path / "precision.nc"
    with jax.enable_x64(x64):
        initial = {
            "Area": jnp.full((6, 7), 2**24, dtype=jnp.float32),
            "index": jnp.array(0, dtype=jnp.int32),
        }

        def advance(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
            trace_modes.append(jax.config.x64_enabled)
            index = state["index"] + 1
            value = jnp.where(index == 1, jnp.float32(1), jnp.float32(-(2**24)))
            return {"Area": jnp.full_like(state["Area"], value), "index": index}

        with OutputManager(path, output) as manager:
            observe, select = output_callbacks(manager, 1)
            final, _, _ = run_timed(initial, advance, 3, observe=observe, select=select)
        assert jax.config.x64_enabled == x64
        assert trace_modes and all(mode == x64 for mode in trace_modes)
        assert final["Area"].dtype == jnp.float32
    record = read_record(path, stream="mean")
    assert record.fields["Area"].dtype == np.float64
    assert record.count == 3 and record.bounds == (0, 3)
    np.testing.assert_array_equal(record.fields["Area"], np.full((2, 3), 1 / 3))


def test_collectors_receive_only_completed_reductions_and_instant_records(
    tmp_path: Path,
) -> None:
    """Collection cost scales with records; each supplied array retains halos."""
    calls = []

    def collect(fields: Any, names: tuple[str, ...]) -> dict[str, Any]:
        assert names == ("Area",)
        assert fields["Area"].shape == (6, 7)
        calls.append(np.asarray(fields["Area"]).copy())
        return {"Area": fields["Area"][2:-2, 2:-2]}

    output = OutputSettings(
        streams=(Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),)
    )
    with OutputManager(tmp_path / "reduced.nc", output, collector=collect) as manager:
        observe, select = output_callbacks(manager, 1)
        run_timed(
            jax.tree.map(jnp.asarray, _fields(0)),
            _advance,
            9,
            observe=observe,
            select=select,
        )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], _fields(0)["Area"] + 1.5)
    np.testing.assert_array_equal(calls[1], _fields(0)["Area"] + 5.5)


def test_scheduled_output_does_not_replay_direct_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public callbacks must dispatch reductions rather than sample replay."""
    output = OutputSettings(
        streams=(Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),)
    )
    with OutputManager(tmp_path / "reduced.nc", output) as manager:

        def reject_sample(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("scheduled output replayed a direct sample")

        monkeypatch.setattr(manager, "sample", reject_sample)
        observe, select = output_callbacks(manager, 1)
        run_timed(
            jax.tree.map(jnp.asarray, _fields(0)),
            _advance,
            5,
            observe=observe,
            select=select,
        )


def test_scheduled_runtime_advances_exactly_requested_steps(tmp_path: Path) -> None:
    calls = []
    output = OutputSettings(
        streams=(
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
            Stream("instant", ("Area",), timedelta(seconds=3)),
        )
    )

    def advance(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        jax.debug.callback(
            lambda value: calls.append(int(value)), state["Area"][0, 0], ordered=True
        )
        return _advance(state)

    with OutputManager(tmp_path / "callbacks.nc", output) as manager:
        observe, select = output_callbacks(manager, 1)
        final, _, _ = run_timed(
            jax.tree.map(jnp.asarray, _fields(0)),
            advance,
            9,
            observe=observe,
            select=select,
        )
    jax.block_until_ready(final)
    jax.effects_barrier()
    assert calls == list(range(9))


def test_invalid_schedule_fails_before_physics_collection_or_file(
    tmp_path: Path,
) -> None:
    calls = []
    output = OutputSettings(
        streams=(Stream("instant", ("Area",), timedelta(seconds=1.5)),)
    )
    path = tmp_path / "invalid.nc"

    def advance(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        jax.debug.callback(lambda: calls.append("physics"), ordered=True)
        return _advance(state)

    with (
        OutputManager(path, output) as manager,
        pytest.raises(ValueError, match="sampling"),
    ):
        observe, select = output_callbacks(manager, 1)
        run_timed(
            jax.tree.map(jnp.asarray, _fields(0)),
            advance,
            5,
            observe=observe,
            select=select,
        )
    jax.effects_barrier()
    assert calls == [] and not path.exists()
