"""Independent offset, validation and output ownership checks for Task 5.

These tests supplement the split-trajectory, dynamic-input and combined-observation
oracles with validation and resource failure cases.
"""

from datetime import timedelta
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.integration_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream
from veris.io.scan import ScheduledObserver
from veris.io.schedule import iter_segments


def _settings() -> OutputSettings:
    return OutputSettings(
        streams=(
            Stream("instant", ("Area",), timedelta(seconds=2)),
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
        )
    )


def test_offset_planning_keeps_absolute_endpoints_and_original_windows() -> None:
    segments = tuple(
        iter_segments(
            _settings(), 1, 5, start_step=3, pending_windows=(None, (0, 4_000_000))
        )
    )
    assert [(s.start, s.stop) for s in segments] == [(3, 4), (4, 6), (6, 8)]
    assert [s.completed for s in segments] == [
        ((1, (0, 4_000_000)),),
        (),
        ((1, (4_000_000, 8_000_000)),),
    ]
    assert [s.instantaneous for s in segments] == [(0,), (0,), (0,)]


def test_sparse_retained_window_flushes_without_a_new_sample() -> None:
    settings = OutputSettings(
        streams=(
            Stream("mean", ("Area",), timedelta(seconds=10), timedelta(seconds=5)),
        )
    )
    segments = tuple(
        iter_segments(settings, 1, 4, start_step=2, pending_windows=((0, 5_000_000),))
    )
    assert [(s.start, s.stop, s.completed) for s in segments] == [
        (2, 5, ((0, (0, 5_000_000)),)),
        (5, 6, ()),
    ]


@pytest.mark.parametrize("start", [True, -1, 1.5])
def test_invalid_absolute_start_fails_eagerly_even_with_no_new_steps(
    start: Any,
) -> None:
    with pytest.raises((ValueError, TypeError), match="start_step"):
        iter_segments(_settings(), 1, 0, start_step=start)


@pytest.mark.parametrize(
    "pending",
    [
        (),
        (None,),
        ((0, 4_000_000), None),
        (None, (0, 3_000_000)),
        (None, (0, 0)),
        (None, (False, 4_000_000)),
        (None, (0, 1_000_000)),
    ],
)
def test_invalid_pending_windows_fail_before_iteration(pending: Any) -> None:
    with pytest.raises(ValueError, match="pending_windows"):
        iter_segments(_settings(), 1, 0, start_step=2, pending_windows=pending)


def test_reduced_resume_is_explicit_and_closed_managers_reject_reuse(
    tmp_path: Path,
) -> None:
    manager = OutputManager(tmp_path / "resume.nc", _settings())
    with pytest.raises(RuntimeError, match="not begun"):
        manager.begin_reduced(resume=True)
    manager.begin_reduced()
    with pytest.raises(RuntimeError, match="already"):
        manager.begin_reduced()
    manager.begin_reduced(resume=True)
    with pytest.raises(RuntimeError, match="reduced"):
        manager.sample({"Area": np.ones((6, 6))}, timedelta(0))
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.begin_reduced(resume=True)
    assert not (tmp_path / "resume.nc").exists()


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_continuation_rejects_changed_schema_without_changing_pending_buffers(
    tmp_path: Path, change: str
) -> None:
    settings = OutputSettings(
        streams=(Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),)
    )
    with OutputManager(tmp_path / "schema.nc", settings) as manager:
        observer, select = output_callbacks(manager, 1, continue_runs=True)
        state = {"Area": jnp.zeros((6, 6), dtype=jnp.float32)}
        final, _, _ = run_timed(
            state,
            lambda value: {"Area": value["Area"] + 1},
            1,
            observe=observer,
            select=select,
        )
        assert isinstance(observer, ScheduledObserver)
        sums, counts, windows = observer.sums, observer.counts, observer.pending_windows
        invalid = {
            "Area": jnp.ones(
                (6, 7) if change == "shape" else (6, 6),
                dtype=jnp.float32 if change == "shape" else jnp.int32,
            )
        }
        with pytest.raises(ValueError, match="shape or dtype"):
            run_timed(invalid, lambda value: value, 1, observe=observer, select=select)
        assert (
            observer.sums is sums
            and observer.counts is counts
            and observer.pending_windows is windows
        )
        assert observer.next_step == 1 and not (tmp_path / "schema.nc").exists()
        resumed, _, _ = run_timed(
            final,
            lambda value: {"Area": value["Area"] + 1},
            3,
            observe=observer,
            select=select,
        )
        np.testing.assert_array_equal(resumed["Area"], 4)


@pytest.mark.parametrize("keyword", ["history_select", "history_sink"])
def test_history_selector_and_sink_are_paired_before_any_observation(
    keyword: str,
) -> None:
    effects = []
    controls: dict[str, Any] = {keyword: lambda value: value}
    with pytest.raises(ValueError, match="together"):
        run_timed(
            jnp.array(0),
            lambda value: value + 1,
            1,
            observe=lambda value, index: effects.append(index),
            **controls,
        )
    assert effects == []


def test_writer_closes_even_when_final_mean_flush_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = OutputManager(
        tmp_path / "failure.nc",
        OutputSettings(
            streams=(
                Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=2)),
            )
        ),
    )
    manager.sample({"Area": np.ones((6, 6))}, timedelta(0))
    manager.sample({"Area": np.ones((6, 6))}, timedelta(seconds=1))
    effects = []
    original_close = manager._writer.close

    def close() -> None:
        effects.append("closed")
        original_close()

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("flush failed")

    monkeypatch.setattr(manager._writer, "append", fail)
    monkeypatch.setattr(manager._writer, "close", close)
    with pytest.raises(RuntimeError, match="flush failed"):
        manager.close(timedelta(seconds=2))
    assert effects == ["closed"] and manager.buffer_nbytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        manager.begin_reduced()


def test_zero_history_selection_uses_physics_precision_for_new_arrays(
    tmp_path: Path,
) -> None:
    """Float64 mean reductions must not change a float32 observer's constants."""
    chunks = []
    with (
        jax.enable_x64(False),
        OutputManager(tmp_path / "empty.nc", _settings()) as manager,
    ):
        observer, select = output_callbacks(manager, 1)
        state = {"Area": jnp.ones((6, 6), dtype=jnp.float32)}
        final, _, _ = run_timed(
            state,
            lambda value: value,
            0,
            observe=observer,
            select=select,
            history_select=lambda value: jnp.array(1.0),
            history_sink=chunks.append,
        )
    assert final is state and len(chunks) == 1
    assert chunks[0].shape == (0,) and chunks[0].dtype == np.dtype("float32")
