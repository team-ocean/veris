"""Reduce half-open scheduled samples alongside one pure numerical trajectory.

Device sums/counts stay outside State. Optional post-step histories use the
existing rollout observer; the default reducer has no scan outputs. Continuation
stores only output buffers and absolute indices, never an old numerical State.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from numbers import Integral
from time import perf_counter
from types import FunctionType
from typing import Any
from weakref import WeakKeyDictionary, ref

import jax
import jax.numpy as jnp

from veris.execution.checkpoint import checkpoint_transition
from veris.execution.rollout import step
from veris.io.calendar import duration_us
from veris.io.configuration import Stream
from veris.io.output import OutputManager
from veris.io.schedule import iter_segments, timestep_us

Carry = tuple[Any, tuple[dict[str, jax.Array], ...], tuple[jax.Array, ...], jax.Array]


@dataclass(frozen=True)
class _FieldSelector:
    """Hashable selected-field contract without host file or observer resources."""

    names: tuple[str, ...]
    from_carry: bool = False

    def __call__(self, source: Any) -> dict[str, Any]:
        """Select the same State/mapping fields, optionally from an auxiliary carry."""
        state = source[0] if self.from_carry else source
        return {
            name: state[name] if isinstance(state, Mapping) else getattr(state, name)
            for name in self.names
        }


class _WeakCallable:
    """Key a live callback by identity without owning its captured resources."""

    __slots__ = ("__weakref__", "_identity", "_reference")

    def __init__(self, callback: Callable[..., Any]) -> None:
        """Accept unhashable callbacks; non-weak-referenceable ones use a fallback."""
        self._identity = id(callback)
        self._reference = ref(callback)

    def __hash__(self) -> int:
        """Hash identity without hashing the callback or its weak reference."""
        return self._identity

    def __eq__(self, other: object) -> bool:
        """Reuse live identities without equating separate expired callbacks."""
        if self is other:
            return True
        if not isinstance(other, _WeakCallable):
            return False
        callback = self._reference()
        return callback is not None and callback is other._reference()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve a strong local reference only while tracing a live callback."""
        callback = self._reference()
        if callback is None:
            raise RuntimeError("scheduled numerical callback is no longer alive")
        return callback(*args, **kwargs)


class ScheduledObserver:
    """Scheduled output callback with optional absolute output continuation."""

    def __init__(
        self,
        manager: OutputManager,
        step_seconds: float,
        *,
        continue_runs: bool = False,
        start_step: int = 0,
        mean_coverage_start_us: int = 0,
        emit_initial_instant: bool = True,
    ) -> None:
        """Bind a schedule without allocating arrays, sampling or opening files."""
        for name, value in (
            ("continue_runs", continue_runs),
            ("emit_initial_instant", emit_initial_instant),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        for name, value in (
            ("start_step", start_step),
            ("mean_coverage_start_us", mean_coverage_start_us),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.manager = manager
        self.step_seconds = step_seconds
        self.step_us = timestep_us(step_seconds)
        self.continue_runs = continue_runs
        self.next_step = int(start_step)
        self.mean_coverage_start_us = int(mean_coverage_start_us)
        # Restart suppresses its checkpoint instant without marking freshly
        # empty mean buffers initialized or disabling time-zero mean sampling.
        self.emit_initial_instant = emit_initial_instant
        self.initialized = False
        self.sums: tuple[dict[str, jax.Array], ...] | None = None
        self.counts: tuple[jax.Array, ...] | None = None
        self.pending_windows: tuple[tuple[int, int] | None, ...] | None = None
        self._signature: tuple[Any, ...] | None = None
        self.names = tuple(
            dict.fromkeys(
                name for s in manager.settings.streams for name in s.variables
            )
        )

    def select(self, state: Any) -> dict[str, Any]:
        """Select configured arrays without changing their spatial layout."""
        return _FieldSelector(self.names)(state)

    def __call__(self, sample: Any, iteration: int) -> None:
        """Support ordinary direct sampling outside the scheduled runner."""
        self.manager.sample(sample, timedelta(microseconds=iteration * self.step_us))

    def release(self) -> None:
        """Detach unfinished device reductions when the host lifecycle closes."""
        self.sums = self.counts = self.pending_windows = self._signature = None


def _field_signature(
    state: Any, streams: tuple[Stream, ...], select: Callable[[Any], Any]
) -> tuple[Any, ...]:
    selected = select(state)
    shape = None
    signature = []
    for name in dict.fromkeys(name for stream in streams for name in stream.variables):
        array = jnp.asarray(selected[name])
        if array.ndim != 2 or array.dtype.kind not in "fiu" or min(array.shape) < 1:
            raise ValueError(
                f"variable {name} must be a nonempty real numeric 2D array"
            )
        if shape is not None and array.shape != shape:
            raise ValueError(f"variable {name} has inconsistent grid shape")
        shape = array.shape
        signature.append((name, array.shape, str(array.dtype)))
    return tuple(signature)


def _initial_carry(
    state: Any, streams: tuple[Stream, ...], select: Callable[[Any], Any]
) -> Carry:
    _field_signature(state, streams, select)
    selected = select(state)
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


def _reduction_transition(
    *,
    advance: Callable[..., Any],
    select: Callable[[Any], Any],
    streams: tuple[Stream, ...],
    intervals: tuple[int, ...],
    sample_initial: bool,
    checkpoint: bool,
    physics_x64: bool,
    binary: bool,
) -> Callable[..., Carry]:
    """Share pre-transition sampling and unchanged physics between executors."""
    transition = checkpoint_transition(advance) if checkpoint else advance

    def body(value: Carry, prescribed: Any = None) -> Carry:
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
            state = transition(state, prescribed) if binary else transition(state)
        return state, tuple(updated_sums), tuple(updated_counts), iteration + 1

    return body


def _reduce_chunk(
    carry: Carry,
    *,
    advance: Callable[..., Any],
    select: Callable[[Any], Any],
    streams: tuple[Stream, ...],
    intervals: tuple[int, ...],
    sample_initial: bool,
    checkpoint: bool,
    physics_x64: bool,
    steps: int,
    inputs: Any = None,
) -> Carry:
    """Execute a history-free scan; retain the established four-item carry."""
    transition = _reduction_transition(
        advance=advance,
        select=select,
        streams=streams,
        intervals=intervals,
        sample_initial=sample_initial,
        checkpoint=checkpoint,
        physics_x64=physics_x64,
        binary=inputs is not None,
    )

    def body(value: Carry, prescribed: Any) -> tuple[Carry, None]:
        return transition(value, prescribed), None

    return jax.lax.scan(body, carry, xs=inputs, length=steps)[0]


def _observed_chunk(
    carry: Carry,
    *,
    history_select: Callable[[Any], Any],
    inputs: Any = None,
    steps: int,
    **settings: Any,
) -> tuple[Carry, Any]:
    """Select post-step values from the same reduction/physics trajectory."""
    transition = _reduction_transition(binary=inputs is not None, **settings)
    physics_x64 = settings["physics_x64"]

    def observe(value: Carry) -> Any:
        with jax.enable_x64(physics_x64):
            return history_select(value[0])

    return step(
        carry, transition, steps, inputs=inputs, observe=observe, checkpoint=False
    )


_scheduled_executors: WeakKeyDictionary[FunctionType, Any] = WeakKeyDictionary()


def _compile_chunk(
    carry: Carry,
    *,
    advance: Callable[..., Any],
    select: Callable[[Any], Any],
    history_select: Callable[[Any], Any] | None,
    inputs: Any,
    **settings: Any,
) -> Any:
    """Reuse default reductions only for the lifetime of a plain transition.

    Weak function keys release their JIT owners when the caller discards the
    transition. Cached partials hold only its weak identity and immutable field
    selection; changing numerical carry/forcing remains dynamic. Arbitrary
    callable objects and history selectors retain the scoped execution path.
    """
    if (
        type(advance) is FunctionType
        and type(select) is _FieldSelector
        and history_select is None
    ):
        execute = _scheduled_executors.get(advance)
        if execute is None:
            execute = jax.jit(
                partial(_reduce_chunk, advance=_WeakCallable(advance)),
                static_argnames=(
                    "select",
                    "streams",
                    "intervals",
                    "sample_initial",
                    "checkpoint",
                    "physics_x64",
                    "steps",
                ),
            )
            _scheduled_executors[advance] = execute
        return execute.lower(carry, select=select, inputs=inputs, **settings).compile()
    # Keep strong, hashable proxies alive only during lowering. Nested scan and
    # checkpoint caches see weak proxies, including for slots/unhashable objects.
    owners = {"advance": partial(advance), "select": partial(select)}
    if history_select is not None:
        owners["history_select"] = partial(history_select)
    callbacks = {name: _WeakCallable(value) for name, value in owners.items()}
    reducer = (
        _reduce_chunk
        if history_select is None
        else partial(_observed_chunk, history_select=callbacks["history_select"])
    )
    execute = jax.jit(
        partial(
            reducer,
            advance=callbacks["advance"],
            select=callbacks["select"],
            **settings,
        )
    )
    return execute.lower(carry, inputs=inputs).compile()


def run_scheduled(
    state: Any,
    advance: Callable[..., Any],
    steps: int,
    observer: ScheduledObserver,
    select: Callable[[Any], Any],
    *,
    chunk_size: int | None,
    checkpoint: bool,
    inputs: Any = None,
    history_select: Callable[[Any], Any] | None = None,
    history_sink: Callable[[Any], None] | None = None,
) -> tuple[Any, float, float]:
    """Compile event shapes, run once and retain only unfinished continuation."""
    owner = getattr(select, "__self__", None)
    if (
        isinstance(owner, ScheduledObserver)
        and getattr(select, "__func__", None) is ScheduledObserver.select
    ):
        select = _FieldSelector(owner.names)
    manager, settings = observer.manager, observer.manager.settings
    start_step = observer.next_step
    cap = (
        (8 if chunk_size is None else chunk_size)
        if history_select is not None
        else chunk_size
    )
    pending = observer.pending_windows if observer.continue_runs else None
    # Planner preflight precedes manager reservation and all file effects.
    segments = iter_segments(
        settings,
        observer.step_seconds,
        steps,
        cap,
        start_step=start_step,
        pending_windows=pending,
    )
    step_us = observer.step_us
    intervals = tuple(
        duration_us(s.sampling_interval) // step_us for s in settings.streams
    )
    physics_x64 = jax.config.x64_enabled

    def select_physics(value: Any) -> Any:
        with jax.enable_x64(physics_x64):
            return select(value)

    signature = _field_signature(state, settings.streams, select_physics)
    if (
        observer.initialized
        and observer.continue_runs
        and signature != observer._signature
    ):
        raise ValueError("continuation output variable shape or dtype changed")
    manager.begin_reduced(resume=observer.continue_runs and observer.initialized)

    def chunk_inputs(segment: Any) -> Any:
        if inputs is None:
            return None
        return jax.tree.map(
            lambda value: value[segment.start - start_step : segment.stop - start_step],
            inputs,
        )

    started = perf_counter()
    with jax.enable_x64():
        if observer.continue_runs and observer.initialized:
            assert observer.sums is not None and observer.counts is not None
            carry: Carry = (
                state,
                observer.sums,
                observer.counts,
                jnp.array(start_step, jnp.int64),
            )
        else:
            carry = _initial_carry(state, settings.streams, select_physics)
            carry = (*carry[:3], jnp.array(start_step, jnp.int64))
        executables = {}
        for segment in segments:
            size = segment.stop - segment.start
            if size not in executables:
                kwargs = {
                    "advance": advance,
                    "select": select,
                    "streams": settings.streams,
                    "intervals": intervals,
                    "sample_initial": settings.sample_initial,
                    "checkpoint": checkpoint,
                    "physics_x64": physics_x64,
                    "steps": size,
                }
                executables[size] = _compile_chunk(
                    carry,
                    inputs=chunk_inputs(segment),
                    history_select=history_select,
                    **kwargs,
                )
        # Empty histories infer shapes with identity only, without physics/forcing.
        empty_history = None
        if not steps and history_select is not None:

            def history_physics(value: Any) -> Any:
                with jax.enable_x64(physics_x64):
                    return history_select(value)

            identity = jax.jit(
                lambda value: step(
                    value, lambda x: x, 0, observe=history_physics, checkpoint=False
                )
            )
            empty_execute = identity.lower(state).compile()
    compilation_seconds = perf_counter() - started
    started = perf_counter()
    if not steps and history_select is not None:
        _, empty_history = empty_execute(state)
        assert history_sink is not None
        history_sink(empty_history)

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

    if (
        settings.sample_initial
        and observer.emit_initial_instant
        and start_step == 0
        and not observer.initialized
    ):
        instant(
            tuple(
                i for i, s in enumerate(settings.streams) if s.period == "instantaneous"
            ),
            0,
        )
    for segment in iter_segments(
        settings,
        observer.step_seconds,
        steps,
        cap,
        start_step=start_step,
        pending_windows=pending,
    ):
        result = executables[segment.stop - segment.start](
            carry, inputs=chunk_inputs(segment)
        )
        if history_select is None:
            carry = result
        else:
            carry, history = result
            assert history_sink is not None
            history_sink(history)
        state, sums, counts, iteration = carry
        sums, counts = list(sums), list(counts)
        with jax.enable_x64():
            for index, (left, right) in segment.completed:
                count = int(counts[index])
                if left >= observer.mean_coverage_start_us and count:
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
    if observer.continue_runs:
        assert carry is not None
        end_step = start_step + steps
        windows = []
        for index, interval in enumerate(intervals):
            if int(carry[2][index]):
                last_sample = ((end_step - 1) // interval) * interval * step_us
                windows.append(
                    manager.clock.window(
                        timedelta(microseconds=last_sample),
                        settings.streams[index].period,
                    )
                )
            else:
                windows.append(None)
        observer.sums, observer.counts = carry[1:3]
        observer.pending_windows = tuple(windows)
        observer.next_step = end_step
        observer._signature = signature
        observer.initialized = True
    return carry[0], compilation_seconds, perf_counter() - started
