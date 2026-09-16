"""Scheduled output rejects invalid arrays and reused managers before tracing."""

from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from veris.integration_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream


def _settings() -> OutputSettings:
    return OutputSettings(
        streams=(Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=2)),)
    )


@pytest.mark.parametrize("dtype", ["complex64", "bool"])
def test_mean_rejects_nonreal_fields_before_casting(tmp_path: Path, dtype: str) -> None:
    path = tmp_path / "invalid.nc"
    traces = []
    state = {"Area": jnp.ones((6, 7), dtype=dtype)}

    def advance(value: dict[str, jax.Array]) -> dict[str, jax.Array]:
        traces.append(True)
        return value

    with OutputManager(path, _settings()) as manager:
        observe, select = output_callbacks(manager, 1)
        with pytest.raises(ValueError, match="real numeric 2D"):
            run_timed(state, advance, 2, observe=observe, select=select)
    assert not traces and not path.exists()


@pytest.mark.parametrize("closed", [False, True])
def test_reused_manager_rejects_before_compiling(tmp_path: Path, closed: bool) -> None:
    traces = []
    state = {"Area": jnp.ones((6, 7))}

    def advance(value: dict[str, jax.Array]) -> dict[str, jax.Array]:
        traces.append(True)
        return value

    with OutputManager(tmp_path / "used.nc", _settings()) as manager:
        manager.sample(state, timedelta(0))
        if closed:
            manager.close()
        observe, select = output_callbacks(manager, 1)
        with pytest.raises(RuntimeError, match="closed|sampled"):
            run_timed(state, advance, 2, observe=observe, select=select)
    assert not traces
