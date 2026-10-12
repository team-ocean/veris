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

from veris.execution.rollout import step
from veris.io.guard import require_host
from veris.io.output import OutputManager
from veris.io.scan import ScheduledObserver, _FieldSelector, run_scheduled

_T = TypeVar("_T")


def run_timed(
    state: _T,
    advance: Callable[..., _T],
    steps: int,
    *,
    observe: Callable[[Any, int], None] | None = None,
    select: Callable[[_T], Any] | None = None,
    chunk_size: int | None = None,
    checkpoint: bool = True,
    inputs: Any = None,
    history_select: Callable[[_T], Any] | None = None,
    history_sink: Callable[[Any], None] | None = None,
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
    for name, callback in (
        ("observe", observe),
        ("select", select),
        ("history_select", history_select),
        ("history_sink", history_sink),
    ):
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
    if (history_select is None) != (history_sink is None):
        raise ValueError("history_select and history_sink must be supplied together")
    if inputs is not None:
        leaves = jax.tree.leaves(inputs)
        if not leaves or any(
            not hasattr(leaf, "shape") or not leaf.shape or leaf.shape[0] != steps
            for leaf in leaves
        ):
            raise ValueError(
                "inputs must contain array leaves with leading length steps"
            )
    if isinstance(observe, ScheduledObserver):
        return run_scheduled(
            state,
            advance,
            steps,
            observe,
            select if select is not None else observe.select,
            chunk_size=chunk_size,
            checkpoint=checkpoint,
            inputs=inputs,
            history_select=history_select,
            history_sink=history_sink,
        )
    chunk_size = 8 if chunk_size is None else int(chunk_size)
    host_select = (
        (select if select is not None else lambda value: value)
        if observe is not None
        else None
    )
    selector = host_select if history_select is None else history_select
    if host_select is not None and history_select is not None:
        selector = lambda value: (host_select(value), history_select(value))
    observed = selector is not None
    length = min(steps, chunk_size) if observed else steps
    lengths = () if steps == 0 else tuple(dict.fromkeys((length, steps % length)))

    def sliced(start: int, size: int) -> Any:
        return (
            None
            if inputs is None
            else jax.tree.map(lambda value: value[start : start + size], inputs)
        )

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
            executables[size] = execute.lower(state, inputs=sliced(0, size)).compile()
    empty_execute = None
    if not steps and history_select is not None:
        execute = jax.jit(
            lambda value: step(
                value, lambda x: x, 0, observe=history_select, checkpoint=False
            )
        )
        empty_execute = execute.lower(state).compile()
    compilation_seconds = perf_counter() - started
    started = perf_counter()
    if observe is not None:
        assert host_select is not None
        observe(host_select(state), 0)
    if not observed:
        if steps:
            state = executables[steps](state, inputs=inputs)
    else:
        for start in range(0, steps, chunk_size):
            size = min(chunk_size, steps - start)
            state, history = executables[size](state, inputs=sliced(start, size))
            host_history, selected_history = history, history
            if host_select is not None and history_select is not None:
                host_history, selected_history = history
            if observe is not None:
                for index in range(size):
                    sample = jax.tree.map(
                        lambda array, index=index: array[index], host_history
                    )
                    observe(sample, start + index + 1)
            if history_sink is not None:
                history_sink(selected_history)
        if empty_execute is not None:
            _, empty_history = empty_execute(state)
            assert history_sink is not None
            history_sink(empty_history)
    jax.block_until_ready(state)
    return state, compilation_seconds, perf_counter() - started


def output_callbacks(
    manager: OutputManager,
    step_seconds: float,
    *,
    continue_runs: bool = False,
    start_step: int = 0,
    mean_coverage_start_us: int = 0,
    emit_initial_instant: bool = True,
) -> tuple[Callable[[Any, int], None] | None, Callable[[Any], Any] | None]:
    """Select the union of configured fields and retain halos for the collector.

    Disabled output returns no callbacks, so the host runner allocates no history.
    The callable carries scheduling metadata for device reductions in run_timed.
    It also supports direct host calls with the manager's ordinary sample API.
    """
    if not manager.settings.enabled:
        return None, None
    observe = ScheduledObserver(
        manager,
        step_seconds,
        continue_runs=continue_runs,
        start_step=start_step,
        mean_coverage_start_us=mean_coverage_start_us,
        emit_initial_instant=emit_initial_instant,
    )
    return observe, _FieldSelector(observe.names)
