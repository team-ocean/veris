"""Host scan chunks preserve selected output fields and compile without steps."""

from datetime import timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.integration_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream, read_record


@pytest.mark.parametrize("observed", [False, True])
def test_compilation_executes_no_warmup_trajectory(observed: bool) -> None:
    """A runtime marker catches duplicate physics work hidden by discarded State."""
    executed = []

    def advance(value: jax.Array) -> jax.Array:
        jax.debug.callback(lambda x: executed.append(int(x)), value, ordered=True)
        return value + 1

    samples = []
    observer = (
        (lambda value, index: samples.append((int(value), index))) if observed else None
    )
    final, compilation, integration = run_timed(
        jnp.array(0),
        advance,
        5,
        chunk_size=2,
        observe=observer,
    )
    assert int(final) == 5
    assert executed == [0, 1, 2, 3, 4], "ERROR compilation executed model steps"
    assert samples == ([(i, i) for i in range(6)] if observed else [])
    assert compilation >= 0 and integration >= 0


def test_chunks_tail_preserve_selected_samples() -> None:
    """Compiling full and tail shapes must not alter the actual observations."""
    samples = []
    initial = {"Area": jnp.zeros((6, 7)), "unused": jnp.ones((6, 7))}
    final, warmup, elapsed = run_timed(
        initial,
        lambda state: {**state, "Area": state["Area"] + 1},
        5,
        chunk_size=2,
        select=lambda state: {"Area": state["Area"]},
        observe=lambda sample, iteration: samples.append((sample, iteration)),
    )
    assert all(set(sample) == {"Area"} for sample, _ in samples)
    assert [iteration for _, iteration in samples] == list(range(6))
    assert all(sample["Area"].shape == (6, 7) for sample, _ in samples)
    assert [int(sample["Area"][0, 0]) for sample, _ in samples] == list(range(6))
    np.testing.assert_array_equal(final["Area"], 5)
    assert warmup >= 0 and elapsed >= 0


def test_disabled_output_allocates_no_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled history executes one full rollout with no observation tree."""
    import veris.integration_output as driver

    original = driver.step
    calls = []

    def traced_step(initial: Any, advance: Any, steps: int, **kwargs: Any) -> Any:
        calls.append((steps, kwargs["observe"]))
        return original(initial, advance, steps, **kwargs)

    monkeypatch.setattr(driver, "step", traced_step)
    with OutputManager(
        tmp_path / "disabled.nc", OutputSettings(enabled=False)
    ) as manager:
        observe, select = output_callbacks(manager, 2.0)
        final, _, _ = run_timed(
            jnp.array(0),
            lambda s: s + 1,
            5,
            observe=observe,
            select=select,
            chunk_size=2,
        )
    assert int(final) == 5
    assert calls == [(5, None)]
    assert not (tmp_path / "disabled.nc").exists()


@pytest.mark.parametrize("steps", [0, 5])
def test_output_times_selection_and_complete_averages(
    tmp_path: Path, steps: int
) -> None:
    """Chunk boundaries do not affect scheduled samples or half-open means."""
    path = tmp_path / "history.nc"
    settings = OutputSettings(
        streams=(
            Stream("instant", ("Area",), timedelta(seconds=2)),
            Stream("mean", ("hIceMean",), timedelta(seconds=1), timedelta(seconds=4)),
        )
    )
    initial = {
        "Area": jnp.zeros((6, 7)),
        "hIceMean": jnp.ones((6, 7)),
        "unused": jnp.ones((6, 7)),
    }
    with OutputManager(path, settings) as manager:
        observe, select = output_callbacks(manager, 1.0)
        final, _, _ = run_timed(
            initial,
            lambda s: {name: value + 1 for name, value in s.items()},
            steps,
            observe=observe,
            select=select,
            chunk_size=2,
        )
    assert int(final["Area"][0, 0]) == steps
    with h5netcdf.File(path) as file:
        np.testing.assert_array_equal(
            file.groups["instant"].variables["time"][:], np.arange(0, steps + 1, 2)
        )
        assert set(file.groups["instant"].variables) >= {"Area"}
        assert "unused" not in file.groups["instant"].variables
        if steps:
            record = read_record(path, stream="mean")
            np.testing.assert_array_equal(record.fields["hIceMean"], 2.5)
            assert record.bounds == (0, 4)
        else:
            assert "mean" not in file.groups
    assert read_record(path, stream="instant").fields["Area"].shape == (2, 3)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_counts_fail_before_host_observation(steps: Any) -> None:
    """Invalid durations never produce an initial host sample."""
    samples = []
    with pytest.raises((TypeError, ValueError), match="steps"):
        run_timed(
            jnp.array(0), lambda s: s + 1, steps, observe=lambda s, i: samples.append(i)
        )
    assert not samples


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_invalid_chunk_size_is_rejected(chunk_size: Any) -> None:
    """A bounded rollout requires a positive static chunk length."""
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        run_timed(jnp.array(0), lambda s: s + 1, 2, chunk_size=chunk_size)


# Checkpoint validation precedes both zero-step and scheduled dispatch.
@pytest.mark.parametrize(
    "steps,scheduled,checkpoint",
    [(0, False, 0), (0, True, 1), (1, False, None), (1, True, "false")],
)
def test_invalid_checkpoint_fails_before_compilation_or_output(
    tmp_path: Path, steps: int, scheduled: bool, checkpoint: Any
) -> None:
    traces = []

    def advance(state: Any) -> Any:
        traces.append(True)
        return state

    path = tmp_path / "invalid.nc"
    with OutputManager(path) as manager:
        observe, select = output_callbacks(manager, 1) if scheduled else (None, None)
        with pytest.raises(TypeError, match="checkpoint"):
            run_timed(
                {},
                advance,
                steps,
                observe=observe,
                select=select,
                checkpoint=checkpoint,
            )
    assert not traces and not path.exists()


@pytest.mark.parametrize("argument", ["advance", "observe", "select"])
def test_zero_steps_validates_callables(argument: str) -> None:
    kwargs: dict[str, Any] = {"advance": lambda s: s, argument: 123}
    with pytest.raises(TypeError, match=argument):
        run_timed(jnp.array(0), steps=0, **kwargs)


@pytest.mark.parametrize("transform", ["jit", "grad", "jvp"])
def test_host_runner_rejects_transforms_before_observation(transform: str) -> None:
    """Differentiated rollouts must not accidentally perform host sampling."""
    import jax

    samples = []

    def objective(value: jax.Array) -> jax.Array:
        final, _, _ = run_timed(
            value,
            lambda s: s * s,
            2,
            observe=lambda state, iteration: samples.append(iteration),
        )
        return final

    with pytest.raises(RuntimeError, match="outside JAX"):
        if transform == "jvp":
            jax.jvp(objective, (jnp.array(1.0),), (jnp.array(1.0),))
        else:
            getattr(jax, transform)(objective)(jnp.array(1.0))
    assert not samples
