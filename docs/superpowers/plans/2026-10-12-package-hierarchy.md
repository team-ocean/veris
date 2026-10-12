# Scientific package hierarchy implementation plan

> **For agentic workers:** Use superpowers:executing-plans for root implementation;
> independent specialized agents audit structure, performance, docs and final quality.
> Root owns one coordinated pytest session and its workers.

**Goal:** Reorganize scientific/functional source ownership while preserving public
interfaces and numerics; reduce full correctness wall time to at most 20 minutes.

**Architecture:** Canonical domain modules plus typed module-object aliases at old
paths. Separate geometry/schema ownership and shared halo operations; unchanged
scientific code moves first, measured harness improvements follow.

**Tech Stack:** Python 3.14, JAX 0.11.2, pytest, coverage, Ruff, ty, Sphinx, setuptools.

**Spec:** `docs/superpowers/specs/2026-10-12-package-hierarchy-design.md`.

## Global constraints

- Work on `jax-only`; one coordinated pytest session at a time; local GPU access approved.
- Existing public APIs, 70-field State, numerical equations/order/tolerances,
  frozen scientific archives and JAX differentiation/sharding contracts survive.
- Full CPU correctness <=1200 s; preserve current 94.46% maintained coverage.
- No maintained module >800 lines, import-time cycles or unrelated features.
- All required checks and full correctness pass before committing.

## Review focus

- Patching a legacy module global must affect canonical callers.
- Old serialized classes must resolve to the same canonical PyTree/schema classes.
- Imports in any supported order must not enumerate devices or change JAX settings.
- Regrouped tests must still execute the same reference/child/ty cases.
- Cache/AD staging optimizations must run real kernels with unchanged numerical checks.

## Task 1: Establish migration contracts

Files: `tests/test_package_compatibility.py`, existing per-module test files;
new schema/halo tests go in `tests/model/` and `tests/grid/` before implementation.
Interfaces: every previous module from the spec maps to its canonical implementation;
root exports and old/new symbols retain identity.

- [x] Save pre-move source, test collection and scientific archive evidence.
- [x] Write tests for all mapped modules/export identity, patched globals, canonical
  owners, old pickle names, cold imports and unchanged root API.
- [x] Run these tests to observe missing canonical packages (RED).

Expected: missing canonical modules are the only migration failures.

## Task 2: Move source ownership incrementally

Files: exact old/new modules in the spec mapping, all maintained import consumers.
Interfaces: same functions/signatures and scalar/array contracts; geometry and array
input types have explicit owners; packed halo removal shared with parallel CLI.

- [x] Move model/grid files, create minimal documented package initializers and
  static typed legacy aliases; rerun corresponding migration/numerical checks.
- [x] Move physics/numerics/execution/I/O/presets; update canonical imports in
  source/tests/benchmarks and active docs; retain CLI paths and old imports tests.
- [x] Extract geometry/input aliases and halo removal with existing numerical tests
  plus RED/GREEN owner and serial/sharded packed-layout contracts.
- [x] Check canonical dependency graph, module sizes and scientific AST equivalence.

Expected: compatibility/canonical imports and affected numerical/reference tests pass.

## Task 3: Regroup tests and complete reproducible documentation

Files: `tests/{model,grid,physics,numerics,execution,io,setups}/`, docs/api/*.rst,
quickstart/reference pages, README.md, DESIGN.md, MANIFEST.in, tests/test_packaging.py.
Interfaces: standalone probes/data stay at tests root; original test cases/assertions
stay collected; docs literal public examples continue to execute installed packages.

- [x] Move matching tests and repair root/reference/probe path resolution only.
- [x] Extend source/wheel archive tests before changing packaging to cover canonical
  packages, aliases and the full benchmark harness; observe RED then GREEN.
- [x] Publish tree/mapping/migration path and update architecture/API references.
- [x] Run fast development checks and compare pre/post collection and archives.

Expected: unchanged existing collection apart from new migration regressions;
strict docs and all source distribution requirements pass.

## Task 4: Meet the full-suite runtime budget

Files: tests/conftest.py, measured child AD probes/test fixture owners only as justified.
Interfaces: unchanged scientific inputs, tolerances, cases and coverage ownership.

- [x] Profile/cache or stage real child derivatives in focused sequential experiments.
- [x] Write regressions for any new cache lifecycle/settings behavior before edits.
- [x] Accept only measured savings; preserve explicit VJP/JVP/FD and primal checks.
- [x] Run full cold-session CPU correctness with coverage, external wall time and
  durations. Compare mapped maintained/scientific executed lines with baseline.
- [x] If >1200 s, trace remaining measured compile/dispatch/fixture overhead and
  iterate; record failed approaches, never substitute a fast/subset success.

Expected: full passing correctness wall <=1200 s with preserved scientific coverage.

## Task 5: Verify, audit and commit

Files: `.github/workflows/tests.yml`, CHANGELOG.md, tests/AUDIT.md, spec/plan.
Interfaces: installed APIs/docs, CPU/GPU execution and scientific evidence.

- [x] Run Ruff lint/format/annotations, ty, pip check, strict Sphinx, fresh wheel/sdist,
  archive contents and clean installed literal public examples.
- [x] Run both local GPU-only tests with CUDA explicitly required.
- [x] Obtain independent whole-change quality/scientific/API review and fix blockers
  with reproducing tests. Recheck full suite when production/harness changes warrant.
- [x] Update changelog with final hierarchy, compatibility, timings, coverage and
  hardware verification; commit passing focused reorganization on `jax-only`.

Expected: all requested gates proven from current source and commit recorded.

## Execution ledger

- 2026-10-12: Read DESIGN, CHANGELOG, AGENTS, source/import/API/test/docs/packaging
  inventory. Baseline validation is recorded; full suite is already planned, so
  omit redundant orientation full/fast run. Current tree is clean except user test
  evidence. Independent read-only structure/performance/docs audits dispatched.
- Workflow ruling: continue in the explicitly designated `jax-only` checkout and
  implement the fully specified autonomous goal; user/developer implementation and
  autonomy instructions take precedence over skill approval/worktree handoffs.
  Defer all commits, including design docs, until full-suite correctness passes.
- 2026-10-12: Tasks 1–3 implemented with 97 migration/ownership passes and unchanged
  original collection/reference hashes. Shared halo, cache and packaging RED/GREEN
  contracts pass. Full integrated CPU timing/coverage is live; source hashes remain
  unchanged during the run. Independent review's two callback regressions have
  reviewed weak-identity/fallback and real-output/weakref test drafts, awaiting
  integration and numerical verification after this process finishes.
- 2026-10-12: Initial serial full aborts at 1491.49 s; isolated failing test passes.
  Weak callback/fallback owners fix reviewed hashability and lifetime regressions;
  85 real output contracts pass. Dynamic-State AD drafts produce modest or negative
  gains and are not integrated. Three-worker module-grouped correctness passes
  all 1394 cases (1392 CPU passes, two expected GPU skips) in 782.90 external s;
  maintained coverage 94.64%, no lost unchanged executed lines. Worker caches are
  private, fresh and inherited by children; affinity is set before collection.
  Review then reproduces two portability issues in the new worker test; adaptive
  one-CPU grouping and portable mocks pass all eight runtime tests on one CPU.
  Both CUDA-only tests and clean installed public examples pass. Final cold full
  rerun on the reviewed source remains the commit gate.

- 2026-10-12: Final cold suite passes all 1395 cases (1393 CPU passes/two GPU
  skips) in 783.11 external seconds, 94.64% maintained coverage, zero unchanged
  coverage losses. Current source hashes, GPU contracts and installed examples
  verified; independent review has no remaining Critical/Important findings.
