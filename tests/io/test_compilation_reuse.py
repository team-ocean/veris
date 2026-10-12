"""Scheduled continuation reuses tracing while executing changing dynamic inputs."""

from datetime import timedelta
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.execution.rollout_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream, read_record, scan


@pytest.fixture
def reducer_traces(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record real reducer traces while leaving all numerical execution intact."""
    traces = []
    original = scan._reduce_chunk

    def recorded(*args: Any, **kwargs: Any) -> Any:
        traces.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scan, "_reduce_chunk", recorded)
    return traces


def test_continued_output_reuses_trace_with_different_forcing(
    tmp_path: Path, reducer_traces: list[int]
) -> None:
    """A second host run advances real state and output without retracing physics."""
    traces = []

    def advance(
        state: dict[str, jax.Array], forcing: jax.Array
    ) -> dict[str, jax.Array]:
        traces.append(state["Area"].shape)
        return {"Area": state["Area"] + forcing}

    path = tmp_path / "continued.nc"
    settings = OutputSettings(
        streams=(Stream("instant", ("Area",), timedelta(seconds=1)),)
    )
    initial = {"Area": jnp.zeros((6, 7), dtype=jnp.float32)}
    with OutputManager(path, settings) as manager:
        observer, selector = output_callbacks(manager, 1.0, continue_runs=True)
        first, _, _ = run_timed(
            initial,
            advance,
            1,
            observe=observer,
            select=selector,
            inputs=jnp.asarray([1.0], dtype=jnp.float32),
        )
        final, _, _ = run_timed(
            first,
            advance,
            1,
            observe=observer,
            select=selector,
            inputs=jnp.asarray([7.0], dtype=jnp.float32),
        )
    np.testing.assert_array_equal(final["Area"], 8)
    record = read_record(path, stream="instant")
    assert record.time == 2.0
    np.testing.assert_array_equal(record.fields["Area"], 8)
    assert traces == [(6, 7)], "ERROR continued scheduled physics retraced"
    assert len(reducer_traces) == 1, "ERROR continued scheduled reduction retraced"


def test_same_field_selection_reuses_trace_without_retaining_file_resources(
    tmp_path: Path,
    reducer_traces: list[int],
) -> None:
    """Independent managers share pure selection while their output stays independent."""
    traces = []

    def advance(
        state: dict[str, jax.Array], forcing: jax.Array
    ) -> dict[str, jax.Array]:
        traces.append(state["Area"].shape)
        return {"Area": state["Area"] + forcing}

    settings = OutputSettings(
        streams=(Stream("instant", ("Area",), timedelta(seconds=1)),)
    )
    for index, increment in enumerate((3.0, 5.0)):
        path = tmp_path / f"independent-{index}.nc"
        initial = {"Area": jnp.zeros((6, 7), dtype=jnp.float32)}
        with OutputManager(path, settings) as manager:
            observer, selector = output_callbacks(manager, 1.0)
            final, _, _ = run_timed(
                initial,
                advance,
                1,
                observe=observer,
                select=selector,
                inputs=jnp.asarray([increment], dtype=jnp.float32),
            )
        np.testing.assert_array_equal(final["Area"], increment)
        np.testing.assert_array_equal(
            read_record(path, stream="instant").fields["Area"], increment
        )
    assert traces == [(6, 7)], (
        "ERROR immutable field selection retraced across managers"
    )
    assert len(reducer_traces) == 1, (
        "ERROR scheduled reduction retraced across managers"
    )
