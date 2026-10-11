# Centralized model lifecycle implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement these
> milestones with TDD and focused independent review. AGENTS.md authorizes the
> full implementation and requests specialized audits. Do not stop at a proposal.

**Goal:** Implement every lifecycle requirement in AGENTS.md.
**Architecture:** A thin Simulation delegates definition, initialization, pure
physics, rollout and output/restart to focused modules. Reuse proven existing
components and preserve setup compatibility.
**Tech Stack:** Python >=3.12, JAX >=0.11.1, NumPy, h5netcdf, JSON, pytest, Sphinx.
**Spec:** docs/superpowers/specs/2026-10-11-model-lifecycle-design.md

## Global constraints

- State remains an array-only PyTree; numerical equations/tolerances unchanged.
- Preserve veris.step rollout, registry defaults and immutable scalar objects.
- Public grid dimensions are global physical cells; local packed halos internal.
- One pytest at a time; full correctness before code commits to jax-only.
- No device initialization/JAX configuration mutations at import time.
- Explicit distributed initialization precedes device enumeration.

## Review focus

- Mapping/array mutation after definition must not alter an experiment (task 1).
- Prescribed heat fields overwrite prior coupling outputs; recursive growth stays
  recursive (tasks 3/4).
- Split runs inside an averaging window retain absolute time and means (task 5).
- Incomplete restart data cannot silently allocate default missing fields (task 6).
- Closed-y custom fields and sharding preserve field-specific policies (tasks 2/7).

## Task 1: Validated experiment definitions

Create veris/experiment.py and veris/forcing.py; tests/test_experiment.py and
tests/test_forcing.py. Export Grid, Execution, Experiment, Forcing alongside step.
Produces immutable host definitions, resolved Configuration/PhysicalConstants,
global/local/storage shape validation and validated forcing with constant/series
precedence. Defaults come from existing registries; no model allocation.

- [x] Tests first: shared registry values, physical precision inheritance,
  equal coupled timesteps, global 12x8 on (3,2) gives local (4,4) and storage
  (24,16), invalid partitions, scalar/spatial/series shapes, exhausted forcing,
  unknown fields/settings and defensive immutability. Expected RED: missing API.
- [x] Implement definitions; focused command: pytest tests/test_experiment.py
  tests/test_forcing.py -q. Expected: all pass.
- [x] Run development fast suite and maintained checks. Update CHANGELOG and
  independent definition review. Full suite and remaining gates precede commit.

## Task 2: Declarative definition and runnable initialization

Create veris/experiment_file.py and veris/model_initialization.py;
tests/test_experiment_file.py and tests/test_model_initialization.py. Add
Experiment.from_file(path, overrides=None), to_dict(), builtin(name, ...).
Consumes task 1 schema and existing setup initializers/ocean adapter/allocator.
Produces initialize_experiment(experiment, mesh=None) -> (State,Configuration,
PhysicalConstants); resolved complete definition metadata.

- [x] RED tests: JSON registry < file < Python precedence, nonuniform prescribed
  geometry, masks, initial fields, derived load/inverse thickness/wind magnitude,
  staggered masks, bad geometry, both boundary modes, all-field builtin initial
  equivalence. Expected: missing APIs.
- [x] Implement JSON schema/serialization and geometry adaptation; preserve exact
  existing builtin fields rather than universally cleaning them.
- [x] Focused 120/120, independent reviews and all precommit gates pass; full CPU
  1032 pass/two GPU skips. Fast omitted because full verification was planned.
  Commit separately on jax-only.

## Task 3: Shared pure timestep and forcing

Create veris/model_physics.py and tests/test_model_physics.py. Produces public
advance(state, configuration, constants, forcing=None, *, physics='coupled')
-> (State,Diagnostics), usable under JIT/grad/JVP/VJP.

- [x] RED all-field/diagnostic oracles against island/run_dyn/run_growth,
  prescribed heat overwrite, separate wind magnitude, spatial forcing halos,
  stage order and closed walls. Add JVP/VJP/FD for initial ice and forcing.
- [x] Compose dynamics_transport/Growth unchanged; preserve recursive growth via
  absent prescribed Qnet/Qsw, and exact legacy wrappers where necessary.
- [x] Focused 35 serial/one two-CPU probe and all precommit gates pass; full CPU
  1068 pass/two GPU skips. Preserved old setup APIs and veris.step.

## Task 4: Host Simulation and successive runs

Create veris/simulation.py and tests/test_simulation.py. Produces
Simulation.initialize(experiment, mesh=None), step(), run(steps=None,
observe=None), RunResult and reliable close/context-manager behavior.

- [ ] RED public coupled define/initialize/run result, run(2)+run(3)==run(5),
  absolute forcing cursor/time, optional histories, zero steps, subsecond exact
  clocks, duration defaults, exhaustion preflight and failed-step cleanup.
- [ ] Reuse pure timestep and veris.step; resources and clocks remain host-only.
- [ ] Focused tests/review and complete precommit gates.

## Task 5: Lifecycle scheduled output and selected input

Extend integration_output.py/io/scan.py/io/output.py with continuation offsets
and pending means; tests/test_simulation_output.py. Add Simulation.load_fields
and snapshot; reuse read_record/update_state/write_snapshot/distributed collector.

- [ ] RED split-window output equals uninterrupted half-open means/times/dtypes,
  configured initial sample, calendar origins, resolved metadata, selected input
  validation, snapshots, no unrequested histories and reliable writer closure.
- [ ] Reuse existing event planner/device reductions; retain pending means across
  successive runs, metadata attributes and absolute clock. Do not duplicate a
  per-timestep output framework.
- [ ] Focused output tests, full correctness and all gates before commit.

## Task 6: Complete serial restart

Create veris/restart.py and tests/test_restart.py; add Simulation.save_restart
and from_restart. Reuse netCDF schema/storage and full State restoration.

- [ ] RED uninterrupted/resumed coupled/dynamics/growth all-field comparison,
  both precisions/boundaries, forcing cursor, exact time/calendar, complete
  config/constants, invalid schema/incomplete records, crossing means discarded,
  absolute resumed schedule and explicit distributed rejection.
- [ ] Implement versioned complete metadata and State validation before allocation;
  require new output file, refresh halos and apply documented averaging policy.
- [ ] Focused tests/review and full precommit gates.

## Task 7: Presets, public sharding and migrated examples

Modify setups/island.py, run_dyn.py, run_growth.py, run_parallel.py; add lifecycle
CPU-sharding subprocess probe/test. Preserve compatibility entry points.

- [ ] RED established example CLI results through Simulation, ocean-geometry
  public example, global/local callback packing, both mesh orientations and
  closed-y output, all-field serial/sharded and JVP/VJP/FD equivalence.
- [ ] Route CLI orchestration through facade; construct only addressable partitions
  for distributed input and retain explicit multiprocess resource initialization.
- [ ] CPU full suite plus available GPU/multihost checks; record unavailable gates.

## Task 8: Documentation, installation and completion audit

Update quickstart/user-guide.rst, API pages, README, add lifecycle migration guide,
update DESIGN and CHANGELOG. Add meaningful public-import/install smoke checks.

- [ ] Run documented coupled quickstart and complete custom ocean example from a
  clean wheel-installed environment outside source import path.
- [ ] Maintained Ruff/format/annotation/ty, fresh build-base wheel/sdist, Sphinx
  -E -W, full pytest correctness/coverage >=80%, available GPU/sharding probes.
- [ ] Independent scientific/API/restart review; fix correctness issues with RED
  regression tests. Record requirement-by-requirement authoritative evidence.
- [ ] Commit focused passing milestones to jax-only; only mark goal complete when
  every AGENTS.md requirement and named delivery artifact is verified.
