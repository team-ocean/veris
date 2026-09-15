"""Run pure scan chunks and replay selected samples into host output managers.

This layer owns timing and I/O only; the bound transition owns the physics.
Observed histories retain storage halos until serial or distributed collectors
remove them. At most ``chunk_size`` selected-field samples reside in each scan
history. Unobserved rollouts retain only the final carry. Use ``veris.step``
directly for differentiated rollouts instead of this host-only timing wrapper.
"""

from collections.abc import Callable, Mapping
from datetime import timedelta
from numbers import Integral
from time import perf_counter
from typing import Any, TypeVar

import jax

from veris import step
from veris.io.guard import require_host
from veris.io.output import OutputManager

_T = TypeVar("_T")


def run_timed(
    state: _T,
    advance: Callable[[_T], _T],
    steps: int,
    *,
    observe: Callable[[Any, int], None] | None = None,
    select: Callable[[_T], Any] | None = None,
    chunk_size: int = 8,
    checkpoint: bool = True,
) -> tuple[_T, float, float]:
    """Warm up scan shapes, then time evolution with optional host observations.

    ``advance`` and ``select`` must be pure and JAX traceable. ``observe`` runs
    on the host for the selected initial sample and every subsequent model step;
    it may perform I/O. With no selector an explicit observer receives the whole
    carry for compatibility. Every used chunk length is warmed up from the
    original carry, and warmup samples never reach the host observer.
    """
    require_host()
    for name, value, minimum in (("steps", steps, 0), ("chunk_size", chunk_size, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    steps, chunk_size = int(steps), int(chunk_size)
    selector = select if observe is not None else None
    if observe is not None and selector is None:
        selector = lambda value: value
    length = min(steps, chunk_size) if observe is not None else steps
    lengths = () if steps == 0 else tuple(dict.fromkeys((length, steps % length)))
    started = perf_counter()
    for size in lengths:
        if size:
            jax.block_until_ready(
                step(state, advance, size, checkpoint=checkpoint, observe=selector)
            )
    warmup_seconds = perf_counter() - started
    started = perf_counter()
    if observe is not None:
        assert selector is not None
        observe(selector(state), 0)
    if observe is None:
        if steps:
            state = step(state, advance, steps, checkpoint=checkpoint, observe=None)
    else:
        for start in range(0, steps, chunk_size):
            size = min(chunk_size, steps - start)
            state, history = step(
                state, advance, size, checkpoint=checkpoint, observe=selector
            )
            for index in range(size):
                sample = jax.tree.map(lambda array, index=index: array[index], history)
                observe(sample, start + index + 1)
    jax.block_until_ready(state)
    return state, warmup_seconds, perf_counter() - started


def output_callbacks(
    manager: OutputManager, step_seconds: float
) -> tuple[Callable[[Any, int], None] | None, Callable[[Any], Any] | None]:
    """Select the union of configured fields and retain halos for the collector.

    Disabled output returns no callbacks, so the host runner allocates no history.
    Sampling decisions and complete-window averaging remain owned by the manager.
    """
    if not manager.settings.enabled:
        return None, None
    names = tuple(
        dict.fromkeys(
            name for stream in manager.settings.streams for name in stream.variables
        )
    )

    def select(state: Any) -> dict[str, Any]:
        """Select arrays without transfers or changes to their spatial layout."""
        return {
            name: state[name] if isinstance(state, Mapping) else getattr(state, name)
            for name in names
        }

    def observe(sample: Any, iteration: int) -> None:
        """Replay a concrete sample at its exact elapsed model time."""
        manager.sample(sample, timedelta(seconds=iteration * step_seconds))

    return observe, select
