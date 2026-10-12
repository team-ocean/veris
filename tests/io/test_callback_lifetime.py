"""Scheduled executors accept pure callables without owning caller resources."""

import gc
import weakref
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.execution.rollout_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream, read_record, scan
from veris.io.scan import ScheduledObserver


@dataclass
class _Advance:
    """Ordinary unhashable callable with no mutation of numerical inputs."""

    increment: float = 2.0

    def __call__(self, state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Advance a prescribed field with a scalar numerical oracle."""
        return {"Area": state["Area"] + self.increment}


@dataclass
class _Select:
    """Unhashable pure selector accepted by the existing callable contract."""

    name: str = "Area"

    def __call__(self, state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Preserve field layout and dtype."""
        return {self.name: state[self.name]}


class _SlottedSelect:
    """A valid callable with neither hashing nor weak-reference support."""

    __slots__ = ()
    __hash__ = None

    def __call__(self, state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Select a configured output field."""
        return {"Area": state["Area"]}


@pytest.mark.parametrize(
    "callback", ["advance", "select", "history_select", "slots", "slots_history"]
)
def test_scheduled_executor_accepts_existing_pure_callable_objects(
    tmp_path: Path, callback: str
) -> None:
    """Hashability and weak-reference support are not public requirements."""
    path = tmp_path / "callables.nc"
    settings = OutputSettings(
        streams=(Stream("instant", ("Area",), timedelta(seconds=1)),)
    )
    history = []
    has_history = callback in {"history_select", "slots_history"}
    history_select = (
        (_SlottedSelect() if callback == "slots_history" else _Select())
        if has_history
        else None
    )
    with OutputManager(path, settings) as manager:
        observer, selector = output_callbacks(manager, 1)
        if callback == "select":
            selector = _Select()
        elif callback == "slots":
            selector = _SlottedSelect()
        advance = (
            _Advance()
            if callback == "advance"
            else lambda state: {"Area": state["Area"] + 2.0}
        )
        final, _, _ = run_timed(
            {"Area": jnp.zeros((6, 7), jnp.float32)},
            advance,
            2,
            observe=observer,
            select=selector,
            history_select=history_select,
            history_sink=history.append if has_history else None,
        )
    np.testing.assert_array_equal(final["Area"], 4)
    np.testing.assert_array_equal(read_record(path, stream="instant").fields["Area"], 4)
    if has_history:
        np.testing.assert_array_equal(
            jnp.concatenate([sample["Area"] for sample in history])[:, 0, 0], [2, 4]
        )


@pytest.mark.parametrize("selection", ["default", "bound", "custom", "cached"])
def test_cached_scheduled_executor_releases_closed_caller_resources(
    tmp_path: Path, selection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collection must succeed without globally clearing useful JAX caches."""
    executors = []
    original_jit = jax.jit

    def tracked_jit(*args: Any, **kwargs: Any) -> Any:
        execute = original_jit(*args, **kwargs)
        if (
            args
            and isinstance(args[0], partial)
            and args[0].func in (scan._reduce_chunk, scan._observed_chunk)
        ):
            executors.append(weakref.ref(execute))
        return execute

    monkeypatch.setattr(scan.jax, "jit", tracked_jit)

    def execute() -> list[Any]:
        settings = OutputSettings(
            streams=(Stream("instant", ("Area",), timedelta(seconds=1)),)
        )
        with OutputManager(tmp_path / "released.nc", settings) as manager:
            observer, _ = output_callbacks(manager, 1)
            assert isinstance(observer, ScheduledObserver)

            def advance(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
                # Pure numerical policy, with deliberate owner references to
                # reveal whether a static callback key extends their lifetime.
                increment = observer.step_seconds
                return {"Area": state["Area"] + increment}

            def custom(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
                assert manager.settings.streams[0].variables == ("Area",)
                return {"Area": state["Area"]}

            select = (
                None
                if selection in {"default", "cached"}
                else observer.select
                if selection == "bound"
                else custom
            )
            history = []
            run_timed(
                {"Area": jnp.zeros((6, 7), jnp.float32)},
                advance,
                1,
                observe=observer,
                select=select,
                history_select=None if selection == "cached" else custom,
                history_sink=None if selection == "cached" else history.append,
            )
            refs = [
                weakref.ref(value) for value in (manager, observer, advance, custom)
            ]
        return refs

    references = execute()
    gc.collect()
    assert all(reference() is None for reference in references), (
        "ERROR scheduled JIT cache retained closed caller resources"
    )
    assert executors, "ERROR scheduled executor has no scoped compilation owner"
    assert all(reference() is None for reference in executors), (
        "ERROR scheduled JIT cache retained expired compilation owners"
    )


class _ScaledObserver(ScheduledObserver):
    """A custom pure selector retains its public subclass behavior."""

    def select(self, state: Any) -> dict[str, Any]:
        """Scale selected fields independently of the evolving numerical State."""
        return {name: 2 * value for name, value in super().select(state).items()}


@pytest.mark.parametrize("explicit", [False, True])
def test_cached_selector_preserves_scheduled_observer_override(
    tmp_path: Path, explicit: bool
) -> None:
    """Only the exact original bound selector may become an immutable key."""
    path = tmp_path / "override.nc"
    settings = OutputSettings(
        streams=(Stream("instant", ("Area",), timedelta(seconds=1)),)
    )
    with OutputManager(path, settings) as manager:
        observer = _ScaledObserver(manager, 1)
        final, _, _ = run_timed(
            {"Area": jnp.zeros((6, 7), jnp.float32)},
            lambda state: {"Area": state["Area"] + 2.0},
            2,
            observe=observer,
            select=observer.select if explicit else None,
        )
    np.testing.assert_array_equal(final["Area"], 4)
    np.testing.assert_array_equal(read_record(path, stream="instant").fields["Area"], 8)
