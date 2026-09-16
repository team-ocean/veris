# Scheduled output reductions and compilation-only preparation

The user approved replacing fixed output chunks with schedule-driven chunks,
device-side running sums/counts, and compilation without executing a warmup
trajectory. Implement this in the existing standalone runners using
`.venv-latest`; preserve pure `veris.step` and its AD interface.

## Required behavior

- No intermediate output: compile one full-length scan, then execute it once.
  Zero steps do not compile or execute physics.
- OutputManager-backed output: use chunks ending at the earliest instantaneous
  sample, mean-window end, final model time, or explicitly supplied chunk cap.
  Calendar boundaries between model times flush at the next model time while
  retaining their exact original bounds in output.
- Retain one float64 sum per requested mean field/stream and a scalar count.
  Carry unfinished means across chunks; never stack timestep field histories.
  Sample the pre-transition state inside each chunk, excluding its right end;
  this preserves half-open windows and initial-sample policy. Instantaneous
  records use the state at a chunk endpoint (and initial state when requested).
- Flush only completed, nonempty windows with nonnegative starting times.
  Discard partial first/final windows. Support concurrent independent streams,
  Gregorian leap/century rules, noleap and 360-day calendars, nonaligned mean
  boundaries, and exact microsecond sampling intervals.
- Preserve model precision and the caller's x64 configuration. Means remain
  float64, including float32 cancellation cases; precision context changes
  must be scoped and must not change tracing of the supplied physics callable.
- Keep storage halos until the collector. Device sums remain sharded. Every
  rank enters collection in the same order; nonwriting ranks create no file.
  Scheduled collectors consume reduced fields and must commute with averaging
  (the maintained halo-removal/gather collector does).
- Keep `output_callbacks`' two-value calling convention. A callable scheduled
  observer identifies OutputManager schedules to `run_timed`. Arbitrary host
  callbacks retain bounded selected histories with a default cap of eight.
  For scheduled output the default cap is absent. No-output ignores the cap.
- Compile every used executable shape via `jax.jit(...).lower(...).compile()`;
  compilation may trace pure callables but must execute no model trajectory,
  device callback, sample collection, or file write. The second returned timing
  value measures compilation; the third measures actual execution and I/O.
- A fresh OutputManager can enter reduced-writing mode exclusively. Direct
  `sample` behavior remains unchanged; mixing direct sampling and a reduced run
  on one manager is rejected. The manager retains lazy writing and exception
  cleanup. The scheduled runner owns window completion in this mode.

## Implementation boundaries

`veris/io/schedule.py` contains pure host segment planning. `Segment` carries
start/stop model indices, completed mean streams with exact microsecond bounds,
and instantaneous streams due at its endpoint. `iter_segments(settings,
step_seconds, steps, max_steps=None)` validates schedules before yielding.

`veris/io/scan.py` contains the scheduled observer and pure accumulator scan,
plus compilation/execution orchestration. `integration_output.py` dispatches
scheduled observers or compiles the existing generic scan path without warmup.

`OutputManager.begin_reduced()` selects exclusive reduced-writing mode.
`write_reduced(name, fields, *, time, bounds, count, mean)` accepts already
averaged or instantaneous halo-bearing fields and writes through its existing
collector/writer. Times/bounds are seconds, matching NetCDFWriter. `close`
closes reduced mode without attempting host sampling/averaging again.

## Verification

Independent recurrences and the existing direct OutputManager path are output
oracles. Verify exact chunk endpoints, values, times, bounds and counts; runtime
callbacks prove exactly N executed transitions; scan/Jaxpr shapes prove no
history axis. Cover simultaneous schedules, varying calendar months, partial
windows, precision, zero steps, disabled output and nonwriter collection.
Run actual multi-device CPU and GPU output examples, check pure AD regressions,
and run the full correctness suite before committing back to `jax-only`.
