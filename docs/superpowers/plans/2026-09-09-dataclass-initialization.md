# Registry initialization implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement tasks with review checkpoints.

**Goal:** Implement frozen Settings, PhysicalConstants and minimal State, allocated from documented registries at initialization.

**Architecture:** Explicit typed frozen dataclasses derive defaults from metadata dictionaries. Kernels consume separate configuration and constants; State is an array-only JAX PyTree. Diagnostics are separate outputs.

**Tech stack:** Python 3.14, JAX, pytest, h5netcdf, Sphinx, Ruff and ty.

**Spec:** `docs/superpowers/specs/2026-09-09-dataclass-initialization-design.md` (user approved).

## Global constraints

- Work on the existing `jax-only` checkout; preserve untracked `test_logs/`.
- Activate `.venv-latest`; the 519-test CPU baseline already passed.
- One pytest process at a time; root coordinates all test execution.
- Keep numerical defaults, oracle data and tolerances unchanged.
- Full correctness suite before commits; no remote push.

## 1. Configuration registries and classes

Files: new `veris/configuration.py`, `veris/physical_constants.py`, metadata support;
`veris/settings.py` and `veris/state.py` now re-export the separated Settings class.
Tests: `tests/test_configuration.py` and physical constants tests.

- [x] Write tests asserting `is_dataclass(Settings())`, frozen mutation failure,
  disjoint registry keys, parity with existing defaults, hashability, scalar
  validation and `replace(Settings(), deltatDyn=600).recip_deltatDyn == 1/600`.
- [x] Run focused tests and record failure before implementation.
- [x] Define namedtuple metadata with default/type/description; generate explicit
  typed source fields whose defaults refer to registries, then maintain source.
- [x] Compute exact dependent values in post-init; retain rounded independent
  physical constants. Reject nonfinite scalar inputs and nonpositive denominators.
- [x] Run focused tests, lint and typing. Audit classification before migration.

## 2. Variable metadata and allocation

Files: `veris/variables.py`, `veris/state.py`, new initializer and diagnostics;
tests: new metadata/allocation tests and migrated `tests/test_state.py`.

- [x] Write tests for field/registry parity, frozen mutation, defaults and dtype,
  center/face dimensions, no configuration leaves and real netCDF round trip.
- [x] Record expected failure, implement metadata and registry-derived allocation.
- [x] Retain the 70 directly consumed fields; verify indirect callers before
  removal. Return output-only stresses and freshwater/salt/heat diagnostics in
  a separate frozen diagnostics container with its own output metadata.
- [x] Register State as a dataclass PyTree and verify replace, jit, JVP and VJP.

## 3. Kernel and protocol migration

Files: maintained physics kernels, `_typing.py`, domain protocols, fixtures,
tests and benchmarks that construct or replace configuration.

- [x] Add production-dataclass mass and growth reference/AD tests using
  `SeaIceMass(state, settings, constants)` and equivalent separated signatures.
- [x] Record failure; replace physical attribute reads with constants and pass
  constants explicitly through nested calls/static JIT arguments.
- [x] Move local empirical coefficients to the appropriate registry; retain
  pure equation factors and stencil weights. Include CESM cloud tables,
  saturation formulas, MITgcm drag law, growth cutoffs and solver iterations.
- [x] Remove legacy combined settings classes/dictionaries after all consumers
  migrate. Update bulk wrappers to keep configuration outside numerical State.
- [ ] Run final unchanged physics oracles and gradient tests after all migrations.
  Earlier focused numerical/gradient runs and the development fast sample passed;
  final full-suite validation is running.

## 4. Initialization and execution

Files: public initializer, `veris/setup/artificial.py`, `veris/set_inits.py`,
`veris/fill_overlap.py`, distributed probes and benchmarks.

- [x] Test complete registry-default allocation and invalid keys/types/shapes.
- [x] Test serial initialization without external mesh module and explicit
  sharded mesh initialization; prevent import-time mutable backend selection.
- [x] Implement explicit initialized execution context and migrate halo calls.
- [x] Migrate artificial island setup and immutable geometry initialization;
  expose diagnostics separately while keeping compiled and Python step parity.
- [ ] Run final coupled masked dynamics/growth and gradients on CPU and GPU.
  Focused CPU coupled and four-device initialized-state/gradient probes pass;
  the final CPU/GPU suites remain the completion gate.

## 5. Documentation and final audit

Files: `doc/conf.py`, registry reference pages, quickstart, artificial example,
`DESIGN.md`, `CHANGELOG.md`, and CI if new check paths are needed.

- [x] Generate tables from SETTINGS, PHYSICALCONSTANTS and VARIABLES; verify
  usable netCDF metadata with real h5netcdf round trips.
- [ ] Rebuild Sphinx after the final registry expansion and verify every current
  registry entry appears in the rendered reference pages.
- [x] Obtain independent code and test-quality review and resolve findings.
- [x] Run maintained Ruff/format/annotation/ty checks and full CPU coverage.
- [x] Run full GPU correctness, distributed probes and dependency checks.
- [x] Audit every AGENTS.md Goal requirement against actual source/results.
- [x] Update progress log; validated changes form the migration commit on `jax-only`.

## Implementation status (2026-09-09)

The architecture and migrations above are implemented. Configuration, metadata,
allocation, protocol and kernel changes have focused test evidence; the
four-device initialized-state step/gradient probe passes. Scenario controls
are registered, hCut is a physical optical coefficient, and nx/ny denote local
interior extents for mesh allocation. Independent review identified these
scope gaps before they were resolved. Final CPU/GPU, documentation, annotation,
coverage and commit gates remain unchecked until their current-source results
are recorded by the coordinating agent. No remote push is planned.

Final validation: CPU 625/625 plus 19/19 final mesh checks; GPU-default full
suite 631/631; maintained coverage 98.41%; Ruff, formatting, annotations, ty,
dependency consistency and warning-free registry documentation pass. All
registry keys verified in rendered HTML. Independent review found no blockers.
