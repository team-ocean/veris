"""Compile and time pure scans with optional scheduled or generic host output.

This layer owns timing and I/O only; the bound transition owns the physics.
Scheduled means retain device sums/counts until their output events. Generic
callbacks retain bounded selected-field histories. Unobserved rollouts retain
only the final carry in a single scan. Use ``veris.step``
directly for differentiated rollouts instead of this host-only timing wrapper.
"""

from collections.abc import Callable
from functools import partial
from numbers import Integral
from time import perf_counter
from typing import Any, TypeVar

import jax

from veris import step
from veris.io.guard import require_host
from veris.io.output import OutputManager
from veris.io.scan import ScheduledObserver, run_scheduled

_T = TypeVar("_T")


def run_timed(
    state: _T,
    advance: Callable[[_T], _T],
    steps: int,
    *,
    observe: Callable[[Any, int], None] | None = None,
    select: Callable[[_T], Any] | None = None,
    chunk_size: int | None = None,
    checkpoint: bool = True,
) -> tuple[_T, float, float]:
    """Compile scan shapes, then time evolution with optional host observations.

    ``advance`` and ``select`` must be pure and JAX traceable. Callbacks returned
    by ``output_callbacks`` use scheduled device reductions, with no default
    chunk cap. An explicit positive ``chunk_size`` caps scheduled segments.
    Arbitrary observers receive each step via histories capped at eight steps
    by default; with no selector they receive the whole carry. Every used chunk
    length is lowered and compiled
    without executing physics. Return final State, compilation time and actual
    integration time (including host observation and output).
    """
    require_host()
    if not callable(advance):
        raise TypeError("advance must be a pure callable")
    for name, callback in (("observe", observe), ("select", select)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None")
    if not isinstance(checkpoint, bool):
        raise TypeError("checkpoint must be a static boolean")
    for name, value, minimum in (("steps", steps, 0), ("chunk_size", chunk_size, 1)):
        if name == "chunk_size" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    steps = int(steps)
    if isinstance(observe, ScheduledObserver):
        return run_scheduled(
            state,
            advance,
            steps,
            observe,
            select if select is not None else observe.select,
            chunk_size=chunk_size,
            checkpoint=checkpoint,
        )
    chunk_size = 8 if chunk_size is None else int(chunk_size)
    selector = select if observe is not None else None
    if observe is not None and selector is None:
        selector = lambda value: value
    length = min(steps, chunk_size) if observe is not None else steps
    lengths = () if steps == 0 else tuple(dict.fromkeys((length, steps % length)))
    started = perf_counter()
    executables = {}
    for size in lengths:
        if size:
            execute = jax.jit(
                partial(
                    step,
                    advance=advance,
                    steps=size,
                    checkpoint=checkpoint,
                    observe=selector,
                )
            )
            executables[size] = execute.lower(state).compile()
    compilation_seconds = perf_counter() - started
    started = perf_counter()
    if observe is not None:
        assert selector is not None
        observe(selector(state), 0)
    if observe is None:
        if steps:
            state = executables[steps](state)
    else:
        for start in range(0, steps, chunk_size):
            size = min(chunk_size, steps - start)
            state, history = executables[size](state)
            for index in range(size):
                sample = jax.tree.map(lambda array, index=index: array[index], history)
                observe(sample, start + index + 1)
    jax.block_until_ready(state)
    return state, compilation_seconds, perf_counter() - started


def output_callbacks(
    manager: OutputManager, step_seconds: float
) -> tuple[Callable[[Any, int], None] | None, Callable[[Any], Any] | None]:
    """Select the union of configured fields and retain halos for the collector.

    Disabled output returns no callbacks, so the host runner allocates no history.
    The callable carries scheduling metadata for device reductions in run_timed.
    It also supports direct host calls with the manager's ordinary sample API.
    """
    if not manager.settings.enabled:
        return None, None
    observe = ScheduledObserver(manager, step_seconds)
    return observe, observe.select
