"""Accumulate scheduled arithmetic means during a pure, history-free scan.

The carry holds numerical State, separate float64 field sums, scalar counts and
an iteration. Samples precede transitions, matching OutputManager's half-open
windows. Host code writes completed windows and instantaneous endpoints only;
spatial halos survive until the configured collector. No output buffers enter
the public State or the differentiated ``veris.step`` API.
"""

from collections.abc import Callable, Mapping
from datetime import timedelta
from functools import partial
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

from veris.io.calendar import duration_us
from veris.io.configuration import Stream
from veris.io.output import OutputManager
from veris.io.schedule import iter_segments, timestep_us

Carry = tuple[Any, tuple[dict[str, jax.Array], ...], tuple[jax.Array, ...], jax.Array]


class ScheduledObserver:
    """Callable compatibility adapter carrying an explicit output schedule."""

    def __init__(self, manager: OutputManager, step_seconds: float) -> None:
        """Bind a manager without sampling, allocating arrays or opening files."""
        self.manager = manager
        self.step_seconds = step_seconds
        self.names = tuple(
            dict.fromkeys(
                name for s in manager.settings.streams for name in s.variables
            )
        )

    def select(self, state: Any) -> dict[str, Any]:
        """Select configured arrays without changing their spatial layout."""
        return {
            name: state[name] if isinstance(state, Mapping) else getattr(state, name)
            for name in self.names
        }

    def __call__(self, sample: Any, iteration: int) -> None:
        """Support ordinary direct sampling outside the scheduled runner."""
        self.manager.sample(
            sample, timedelta(microseconds=iteration * timestep_us(self.step_seconds))
        )


def _initial_carry(
    state: Any, streams: tuple[Stream, ...], select: Callable[[Any], Any]
) -> Carry:
    selected = select(state)
    shape = None
    for name in dict.fromkeys(name for stream in streams for name in stream.variables):
        array = jnp.asarray(selected[name])
        if array.ndim != 2 or array.dtype.kind not in "fiu" or min(array.shape) < 1:
            raise ValueError(
                f"variable {name} must be a nonempty real numeric 2D array"
            )
        if shape is not None and array.shape != shape:
            raise ValueError(f"variable {name} has inconsistent grid shape")
        shape = array.shape
    sums = tuple(
        {
            name: jnp.zeros_like(selected[name], dtype=jnp.float64)
            for name in stream.variables
        }
        if stream.period != "instantaneous"
        else {}
        for stream in streams
    )
    return (
        state,
        sums,
        tuple(jnp.array(0, jnp.int64) for _ in streams),
        jnp.array(0, jnp.int64),
    )


def _reduce_chunk(
    carry: Carry,
    *,
    advance: Callable[[Any], Any],
    select: Callable[[Any], Any],
    streams: tuple[Stream, ...],
    intervals: tuple[int, ...],
    sample_initial: bool,
    checkpoint: bool,
    physics_x64: bool,
    steps: int,
) -> Carry:
    transition = jax.checkpoint(advance) if checkpoint else advance

    def body(value: Carry, unused: None) -> tuple[Carry, None]:
        state, sums, counts, iteration = value
        with jax.enable_x64(physics_x64):
            selected = select(state)
        updated_sums, updated_counts = [], []
        for stream, interval, total, count in zip(
            streams, intervals, sums, counts, strict=True
        ):
            if stream.period != "instantaneous":
                due = (iteration % interval == 0) & (sample_initial | (iteration > 0))

                def accumulate(
                    pair: tuple[dict[str, jax.Array], jax.Array],
                ) -> tuple[dict[str, jax.Array], jax.Array]:
                    fields, number = pair
                    return {
                        name: array + selected[name].astype(jnp.float64)
                        for name, array in fields.items()
                    }, number + 1

                total, count = jax.lax.cond(
                    due, accumulate, lambda pair: pair, (total, count)
                )
            updated_sums.append(total)
            updated_counts.append(count)
        with jax.enable_x64(physics_x64):
            state = transition(state)
        return (state, tuple(updated_sums), tuple(updated_counts), iteration + 1), None

    return jax.lax.scan(body, carry, xs=None, length=steps)[0]


def run_scheduled(
    state: Any,
    advance: Callable[[Any], Any],
    steps: int,
    observer: ScheduledObserver,
    select: Callable[[Any], Any],
    *,
    chunk_size: int | None,
    checkpoint: bool,
) -> tuple[Any, float, float]:
    """Compile each event length, then run once and write due reduced records."""
    manager = observer.manager
    settings = manager.settings
    segments = iter_segments(settings, observer.step_seconds, steps, chunk_size)
    step_us = timestep_us(observer.step_seconds)
    intervals = tuple(
        duration_us(s.sampling_interval) // step_us for s in settings.streams
    )
    manager.begin_reduced()
    physics_x64 = jax.config.x64_enabled

    def select_physics(value: Any) -> Any:
        with jax.enable_x64(physics_x64):
            return select(value)

    started = perf_counter()
    with jax.enable_x64():
        carry = _initial_carry(state, settings.streams, select_physics)
        executables = {}
        for segment in segments:
            size = segment.stop - segment.start
            if size not in executables:
                execute = jax.jit(
                    partial(
                        _reduce_chunk,
                        advance=advance,
                        select=select_physics,
                        streams=settings.streams,
                        intervals=intervals,
                        sample_initial=settings.sample_initial,
                        checkpoint=checkpoint,
                        physics_x64=physics_x64,
                        steps=size,
                    )
                )
                executables[size] = execute.lower(carry).compile()
    compilation_seconds = perf_counter() - started
    started = perf_counter()

    def instant(indices: tuple[int, ...], iteration: int) -> None:
        if not indices:
            return
        fields = select_physics(carry[0])
        time = iteration * step_us / 1e6
        for index in indices:
            stream = settings.streams[index]
            manager.write_reduced(
                stream.name,
                {name: fields[name] for name in stream.variables},
                time=time,
                bounds=(time, time),
                count=1,
                mean=False,
            )

    if settings.sample_initial:
        instant(
            tuple(
                i for i, s in enumerate(settings.streams) if s.period == "instantaneous"
            ),
            0,
        )
    for segment in iter_segments(settings, observer.step_seconds, steps, chunk_size):
        carry = executables[segment.stop - segment.start](carry)
        state, sums, counts, iteration = carry
        sums, counts = list(sums), list(counts)
        with jax.enable_x64():
            for index, (left, right) in segment.completed:
                count = int(counts[index])
                if left >= 0 and count:
                    manager.write_reduced(
                        settings.streams[index].name,
                        {name: value / count for name, value in sums[index].items()},
                        time=(left + right) / 2e6,
                        bounds=(left / 1e6, right / 1e6),
                        count=count,
                        mean=True,
                    )
                sums[index] = jax.tree.map(jnp.zeros_like, sums[index])
                counts[index] = jnp.zeros_like(counts[index])
        carry = state, tuple(sums), tuple(counts), iteration
        instant(segment.instantaneous, segment.stop)
    jax.block_until_ready(carry)
    return carry[0], compilation_seconds, perf_counter() - started
