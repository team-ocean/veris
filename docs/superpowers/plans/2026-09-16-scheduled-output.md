# Scheduled output implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or
> superpowers:executing-plans task by task. Root owns all pytest scheduling.

**Goal:** Schedule chunks by output events, accumulate means on device without
histories, and compile without executing duplicate trajectories.

**Architecture:** A pure host planner yields event-aligned segments. A pure
scan carries State and separate sum/count buffers; the host writes only due
records. The generic callback path remains compatible and uses compilation-only
preparation, as does the no-output single-scan path.

**Tech Stack:** Existing JAX, NumPy, h5netcdf, pytest, Ruff, ty and Sphinx.

**Spec:** `docs/superpowers/specs/2026-09-16-scheduled-output-design.md`

## Global constraints

- Activate `.venv-latest`; work on a branch from `jax-only`.
- Preserve half-open arithmetic sample means, float64 accumulation, initial
  sampling choices and complete-window-only output.
- Do not add output buffers or schedules to State; keep I/O outside AD.
- One pytest instance at a time; full correctness before commits. Root runs
  tests for independently edited agent files and reports red/green evidence.

## Task 1: Compilation-only preparation

Files: `veris/integration_output.py`, `tests/test_rollout_output.py`.

- [x] Add a runtime callback regression before implementation:
  ```python
  calls = []
  def advance(value):
      jax.debug.callback(lambda x: calls.append(int(x)), value, ordered=True)
      return value + 1
  result, _, _ = run_timed(jnp.array(0), advance, 5)
  assert calls == [0, 1, 2, 3, 4]
  ```
- [x] Run the regression; confirm the old warmup duplicates the callbacks.
- [x] Compile fixed-length closures without evaluating them:
  ```python
  execute = jax.jit(lambda value: step(value, advance, size,
                                     checkpoint=checkpoint, observe=selector))
  executable = execute.lower(state).compile()
  ```
  Cache by length for the run and invoke only for the actual trajectory.
- [x] Update obsolete tests explicitly: lower traces each shape once, while
  runtime observations and transitions occur only in the actual integration.
  Test no-output, generic callbacks with a tail, zero steps and invalid caps.

## Task 2: Calendar segment planner

Files: new `veris/io/schedule.py`, new `tests/test_output_schedule.py`.

Interface: frozen `Segment(start, stop, completed, instantaneous)`;
`iter_segments(settings, step_seconds, steps, max_steps=None)` yields segments.
`completed` is a tuple of `(stream_index, (left_us, right_us))`; instantaneous
is a tuple of stream indices due at stop. Validate integer counts, positive
microsecond timestep, sampling intervals divisible by timestep, and calendar
range before yielding any segment.

- [x] Write tests before the module: periods 4s/6s, instant every 5s and dt=1s
  give stops `[4, 5, 6, 8, 10, 12, 13]` for 13 steps.
- [x] Add nonaligned 2.5s boundaries, cap/tail, zero steps and incompatible
  sampling tests; calendar month lengths 29/28/30 for leap/noleap/360_day.
- [x] Root runs failing tests; implement event iteration using exact integer
  microseconds and `Calendar.window`. Round only execution endpoints upward:
  ```python
  boundary_step = (right_us + step_us - 1) // step_us
  stop = min(final_step, boundary_step, next_instant_step)
  ```
- [x] Root runs planner tests and reviews omitted empty/partial windows.

## Task 3: Reduced writer lifecycle

Files: `veris/io/output.py`, new `tests/test_output_reduced.py`.

Interface: `begin_reduced()` rejects an already sampled/closed manager;
`write_reduced(name, fields, *, time, bounds, count, mean)` writes already
reduced halo-bearing fields through the configured collector. Direct sampling
and reduced writing are exclusive. Reduced close does not flush host sums.

- [x] Test real h5netcdf records for means and instantaneous fields, collectors
  receiving halos once, nonwriter None, disabled mode and illegal mode mixing.
- [x] Root observes failures; implement the narrow writer/lifecycle interface.
- [x] Keep direct `sample` preflight/clock/averaging semantics intact; run old
  output tests with new lifecycle cases.

## Task 4: Streaming reductions and dispatcher

Files: new `veris/io/scan.py`, new `tests/test_output_scan.py`,
`veris/integration_output.py`.

- [x] Write output oracle tests comparing direct samples of `x(t)=t` to
  scheduled runs for multiple streams, sparse samples, nonaligned windows,
  all calendars, initial exclusion, zero steps and incomplete windows.
- [x] Test float32 `[2**24, 1, -2**24]` yields mean `1/3` with float64 sums.
- [x] Add a callable scheduled observer returned by `output_callbacks`;
  route it to the streaming runner without changing existing call sites.
- [x] Implement the pure carry `(state, stream_sums, counts, iteration)` with
  `scan(..., xs=None, length=size)` and `ys=None`. Sample before advancing,
  gated by cadence and initial policy. Carry sums across unrelated events;
  reset only streams whose windows finish. Do not allocate a time index array.
- [x] Compile unique segment lengths before integration; preserve the physics
  x64 context while accumulating float64 sums in a scoped context.
- [x] At host endpoints, write complete nonempty mean buffers and due instant
  fields, preserving exact bounds/counts and the configured collector.
- [x] Assert runtime callbacks execute exactly N steps. Inspect scan carry and
  result shapes to prove buffer size is independent of timestep count.

## Task 5: Integration, documentation and final verification

Files: maintained runner timing labels/docstrings, `docs/reference/integration.rst`,
`docs/reference/setups/reference-cases.rst`, tests/probes, `CHANGELOG.md`.

- [x] Change timing labels from warmup to compilation and document schedule
  chunks, memory bounds, generic callback cap and reduced collector contract.
- [x] Run actual sharded CPU/GPU scenarios, compare records with serial/direct
  manager output and verify no-output runtime counts and pure AD behavior.
- [x] Obtain independent code and test-quality review; fix reproduced defects.
- [x] Run maintained Ruff/format/annotation/ty, Sphinx, full correctness/coverage
  and GPU-only validation. Record source hashes and exact results.
- [x] Update this plan and CHANGELOG; commit tested work back to `jax-only`.
