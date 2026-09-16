"""Calendar sampling and bounded-memory arithmetic means on the host.

Each mean stream holds only a running sum per selected field and a sample count.
Windows are half-open; boundary samples belong to the new window. This is an
arithmetic sample mean, not a time-weighted quadrature. No numerical State fields
or differentiated integration graphs contain the manager.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from veris._typing import State
from veris.io.calendar import Calendar, duration_us
from veris.io.configuration import OutputSettings, Stream
from veris.io.guard import under_transform
from veris.io.storage import (
    ArrayFields,
    Collector,
    NetCDFWriter,
    selected_fields,
    storage_fields,
)


@dataclass
class _Accumulator:
    stream: Stream
    next_sample: int
    window: tuple[int, int] | None = None
    sums: dict[str, NDArray[Any]] = field(default_factory=dict)
    count: int = 0


class OutputManager:
    """Sample after concrete model steps; disable explicitly around AD loops.

    ``collector`` optionally gathers selected arrays collectively and returns
    physical cells only on the writing rank, with storage halos removed.
    All ranks must call sample with identical schedules when using collectives.
    """

    def __init__(
        self,
        path: str | Path,
        settings: OutputSettings | None = None,
        *,
        collector: Collector | None = None,
    ) -> None:
        settings = OutputSettings() if settings is None else settings
        self.settings = settings
        self.clock = Calendar(settings.start, settings.calendar)
        self._writer = NetCDFWriter(
            path, calendar=self.clock.name, units=self.clock.units
        )
        self._collector = collector
        self._streams = [
            _Accumulator(
                s, 0 if settings.sample_initial else duration_us(s.sampling_interval)
            )
            for s in settings.streams
        ]
        self._last: int | None = None
        self._closed = False
        self._reduced = False

    def __enter__(self) -> Self:
        """Return the lazy output manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush on success; discard pending averages after an integration error."""
        if exc_type is None:
            self.close()
        elif self.settings.enabled and not under_transform():
            self._writer.close()
            self._closed = True
            for stream in self._streams:
                stream.sums.clear()
                stream.count = 0

    @property
    def buffer_nbytes(self) -> int:
        """Current array bytes held for means, independent of sample count."""
        return sum(
            array.nbytes for stream in self._streams for array in stream.sums.values()
        )

    def begin_reduced(self) -> None:
        """Reserve a fresh manager for records reduced by the scheduled runner.

        Reduced writing and direct sampling are exclusive. Disabled output and
        transformed calls remain inert, as they do for :meth:`sample`.
        """
        if not self.settings.enabled or under_transform():
            return
        if self._closed:
            raise RuntimeError("output manager is closed")
        if self._reduced:
            raise RuntimeError("output manager is already in reduced mode")
        if self._last is not None:
            raise RuntimeError("output manager has already sampled directly")
        self._reduced = True

    def write_reduced(
        self,
        name: str,
        fields: ArrayFields,
        *,
        time: float,
        bounds: tuple[float, float],
        count: int,
        mean: bool,
    ) -> None:
        """Collect and write an already reduced, halo-bearing stream record.

        Times and bounds are seconds. The scheduled runner owns sampling and
        window completion; means are already divided by their sample count.
        Collectors must commute with averaging and remove storage halos. Every
        rank must enter collection in the same order, including nonwriters.
        """
        if not self.settings.enabled or under_transform():
            return
        if self._closed:
            raise RuntimeError("output manager is closed")
        if not self._reduced:
            raise RuntimeError("begin_reduced must select reduced writing first")
        stream = next((s for s in self.settings.streams if s.name == name), None)
        if stream is None:
            raise ValueError(f"unknown output stream {name}")
        if mean != (stream.period != "instantaneous"):
            raise ValueError("record kind does not match configured stream")
        if self._collector is None:
            arrays = storage_fields(fields, stream.variables)
        else:
            collected = self._collector(fields, stream.variables)
            if collected is None:
                return
            arrays = selected_fields(collected, stream.variables)
        self._writer.append(
            name, arrays, time=time, bounds=bounds, count=count, mean=mean
        )

    def _flush(self, accumulator: _Accumulator, end: int) -> None:
        if accumulator.window is None or not accumulator.count:
            return
        left, right = accumulator.window
        end = min(right, end)
        if accumulator.sums and left >= 0 and end == right:
            means = {
                name: value / accumulator.count
                for name, value in accumulator.sums.items()
            }
            self._writer.append(
                accumulator.stream.name,
                means,
                time=(left + end) / 2e6,
                bounds=(left / 1e6, end / 1e6),
                count=accumulator.count,
                mean=True,
            )
        accumulator.sums.clear()
        accumulator.count = 0
        accumulator.window = None

    def sample(self, state: State | ArrayFields, elapsed: timedelta) -> None:
        """Process an increasing model time; sample only at exact scheduled times.

        Calls during transformations or in disabled mode have no effects, even
        for constant arrays. Missed samples fail before advancing any stream.
        Non-sampling calls still flush crossed calendar boundaries.
        """
        if not self.settings.enabled or under_transform():
            return
        if self._closed:
            raise RuntimeError("output manager is closed")
        if self._reduced:
            raise RuntimeError("direct sampling is unavailable in reduced mode")
        now = duration_us(elapsed)
        if now < 0 or (self._last is not None and now <= self._last):
            raise ValueError(
                "elapsed model time must be nonnegative and strictly increasing"
            )
        self.clock.date_at(elapsed)
        for accumulator in self._streams:
            if now > accumulator.next_sample:
                raise ValueError(
                    f"missed sample for stream {accumulator.stream.name} at {accumulator.next_sample / 1e6} seconds"
                )
        due = [a for a in self._streams if now == a.next_sample]
        names = tuple(dict.fromkeys(name for a in due for name in a.stream.variables))
        arrays: dict[str, NDArray[Any]] = {}
        if names:
            if self._collector is None:
                arrays = storage_fields(state, names)
            else:
                collected = self._collector(state, names)
                if collected is not None:
                    arrays = selected_fields(collected, names)
        # Preflight shapes and new windows before any record or accumulator mutation.
        windows = {
            a.stream.name: self.clock.window(elapsed, a.stream.period)
            for a in due
            if a.stream.period != "instantaneous"
        }
        for accumulator in due:
            for name, value in accumulator.sums.items():
                if name in arrays and arrays[name].shape != value.shape:
                    raise ValueError(f"variable {name} changed shape")
        if arrays:
            for accumulator in due:
                stream = accumulator.stream
                self._writer.validate_schema(
                    stream.name,
                    {name: arrays[name] for name in stream.variables},
                    mean=stream.period != "instantaneous",
                    mean_samples=stream.period != "instantaneous",
                    time=now / 1e6 if stream.period == "instantaneous" else None,
                )
        for accumulator in self._streams:
            if accumulator.window is not None and now >= accumulator.window[1]:
                self._flush(accumulator, accumulator.window[1])
            if accumulator not in due:
                continue
            stream = accumulator.stream
            if stream.period == "instantaneous":
                if arrays:
                    self._writer.append(
                        stream.name,
                        {n: arrays[n] for n in stream.variables},
                        time=now / 1e6,
                        bounds=(now / 1e6, now / 1e6),
                    )
            else:
                accumulator.window = windows[stream.name]
                for name in stream.variables:
                    if not arrays:
                        break
                    if name not in accumulator.sums:
                        accumulator.sums[name] = np.array(
                            arrays[name], dtype=np.float64, copy=True
                        )
                    else:
                        accumulator.sums[name] += arrays[name]
                accumulator.count += 1
            accumulator.next_sample += duration_us(stream.sampling_interval)
        self._last = now

    def close(self, elapsed: timedelta | None = None) -> None:
        """Finish at the last observed time or an explicit final boundary.

        A close at the next sample time ends the half-open interval without
        requiring a sample at its right endpoint. Later closes would skip samples
        and are rejected. Incomplete first and last averaging windows are
        discarded. Repeated closes are harmless.
        """
        if not self.settings.enabled or under_transform() or self._closed:
            return
        if self._reduced:
            self._writer.close()
            self._closed = True
            return
        end = self._last if elapsed is None else duration_us(elapsed)
        if end is not None:
            if end < 0 or (self._last is not None and end < self._last):
                raise ValueError("close time cannot precede the last model time")
            self.clock.date_at(timedelta(microseconds=end))
            if any(end > a.next_sample for a in self._streams):
                raise ValueError("close time would skip a missed sample")
            for accumulator in self._streams:
                self._flush(accumulator, end)
        self._writer.close()
        self._closed = True
