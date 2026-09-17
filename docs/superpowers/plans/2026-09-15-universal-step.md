# Universal Step Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task, with independent test and code review.

**Goal:** Provide one scan-based rollout API, optional AD checkpointing, and verified integration of every setup.

**Architecture:** A pure generic scan driver uses existing setup transitions. A separate host runner uses bounded observation chunks for netCDF sampling. Diagnostics are auxiliary scan outputs, never carry fields.

**Tech Stack:** Python, JAX, pytest, h5netcdf, Sphinx.

**Spec:** docs/superpowers/specs/2026-09-15-universal-step-design.md (approved by user).

## Global constraints

- Preserve setup physics and static configuration/constants.
- Preserve CPU/GPU, sharding, halo and host-only I/O boundaries.
- One pytest process at a time; root schedules all tests.
- Full correctness/coverage suite before commits; integrate on jax-only.
- Work in current checkout per existing project workflow; no unrelated edits.

## 1. Pure rollout (root)

Files: veris/integration.py, veris/__init__.py, tests/test_integration.py.

Interface: `step(initial, advance, steps, *, checkpoint=True, inputs=None,
observe=None, has_aux=False)`. No observer returns final carry. Observer returns
`(final_carry, stacked_observations)`. `advance(carry)` or
`advance(carry, input_slice)` returns new carry, or `(carry, aux)` when has_aux.
Observers accept new carry, or `(new_carry, aux)` in auxiliary mode.

- [x] Write tests for hand-derived recurrence, zero/one/many steps, forcing
  PyTrees, observation selection, auxiliary outputs, count/shape validation,
  JIT, JVP/VJP and finite differences; inspect scan/remat in actual JAX program.
- [x] Run tests before module exists and confirm missing public API failure.
- [x] Implement static-argument compiled scan with checkpointed body. Reject
  negative/nonintegral counts and forcing with incorrect leading length.
- [x] Run focused tests and resolve discrepancies without tolerance relaxation.

Example independent recurrence test:
```python
result, history = step(jnp.array(1.), lambda s: 2*s, 3, observe=lambda s: s)
np.testing.assert_array_equal(history, [2., 4., 8.])
assert result == 8
```
Zero-step scans trace pure transitions/observers to infer empty output shapes;
no numerical timestep or host side effect is performed.

## 2. Setup and AD tests (validation agent)

Files: tests/test_rollout_setups.py, tests/rollout_parallel_probe.py.

- [x] Write setup regression tests. Compare all State fields
  against explicit setup steps, including island changing cooling and
  ocean geometry, dynamics and growth, float32/64.
- [x] Add nonuniform spatial weighted JVP/VJP/FD objectives for each setup and
  checkpoint on/off. Verify State and forcing sensitivity and diagnostics.
- [x] Add real sharded scan and AD probe, replicated scalar cooling input, and
  explicit serial equivalence. Root runs focused CPU and GPU validations.

## 3. Host integration (driver agent)

Files: veris/integration_output.py, veris/setups/run_dyn.py,
veris/setups/run_growth.py, veris/setups/run_parallel.py,
tests/test_rollout_output.py, tests/test_parallel_case.py.

Interface: host runner uses the public step API with pure selected-field
observation and replays samples outside transformations. No output means no
history allocation. Retain halos in sampled arrays until existing collectors.

- [x] Add failing tests for multi-chunk evolution, tail, selected-field history,
  exact times, averaging, zero steps, and parallel warmup correctness.
- [x] Replace Python physics loops with scans; update old timing-side-effect
  assertions explicitly since advance must now be pure and traceable.
- [x] Warm up every executed scan shape (full chunk and tail) from original
  State before timed integration. Never reuse advanced warmup State.
- [x] Execute focused tests under root scheduling and all actual setup CLIs.

## 4. Documentation, review and final verification (root + reviewer)

- [x] Document API/examples for all setups, forcing, auxiliary diagnostics,
  checkpoint memory tradeoff, sharding, zero-step tracing and host output.
- [x] Independent review of implementation and tests against approved spec.
- [x] Run focused development checks; full CPU correctness/coverage and targeted
  GPU/sharded/AD validations before commit. Run Ruff, formatting, annotation,
  ty and Sphinx checks using existing maintained scope.
- [x] Record exact evidence in CHANGELOG; integrate verified work on jax-only.

## Execution ledger

- User approved design. Root owns generic integration and pytest scheduling;
  independent agents own setup validation and host driver changes.
- Review resolution: add has_aux protocol to preserve Diagnostics; zero-step
  output uses abstract tracing; warmup covers both chunk and tail shapes.

- Focused validation: core/host/CLI 53 passed plus one GPU-only skip; repair and
  float32 serial/four-device AD 29 passed. Two-GPU float64 coupled/dynamics
  forward and AD passed. Full CPU suite and float32 GPU probe active.
- Full-suite inputs frozen in test_logs/scan-source-hashes.json; no production,
  test or rendered-document source edits during final verification.

- Final verification: CPU 943 passed plus one GPU-only skip; 18 GPU regression
  checks including that GPU-only test passed. Two-GPU sharded probes pass both
  precisions; nine-step GPU netCDF CLIs pass. SLURM job 65267175 completed exit0
  and two-process output matches serial within 5.56e-17. Maintained coverage
  96.94%, whole-package 84.87%; static/doc gates and source hashes pass.
