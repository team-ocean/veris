# Test runtime reduction implementation plan

> **For agentic workers:** Use superpowers:executing-plans for the harness changes and a separate code-quality review. Root owns every pytest run.

**Goal:** Reduce full-suite runtime and memory, remove redundant tests, and use float32 for precision-independent operational contracts while preserving scientific coverage.

**Architecture:** Change test ownership, fixtures and repeated reference execution. Keep production kernels, equations, tolerances, full correctness selection and coverage exclusions unchanged. Measure the current suite and the final suite with the same environment, recording per-file executed lines, wall time and peak process RSS. Coverage 7.16.0's Python 3.14 sysmon core cannot provide complete dynamic test contexts; contract ownership requires static review in addition to the authoritative global line sets.

**Tech stack:** Python 3.14, pytest, pytest-cov, JAX CPU, netCDF, existing `.venv-latest`.

**Spec:** User's active test-runtime objective; numerical/precision and test workflow constraints in `DESIGN.md` and `AGENTS.md`.

## Global constraints

- Work on `jax-only`; one pytest instance at a time.
- Keep exact restarts, all-field references, invalid-input cases, boundaries, JIT, AD and sharding contracts.
- Float64 remains for tight equation references and finite differences; both dtypes remain where dtype is the contract.
- Do not relax tolerances, add skips, remove branches, weaken full-suite selection or add coverage exclusions.
- Full correctness and CI-scope lint/type/docs/package checks precede any commit.

## Review focus

- Mutating one corrupt checkpoint must never contaminate another test's template.
- Shared serial references must remain independent of each real parallel trajectory.
- Checkpoint positions zero, interior and completed must retain distinct clock/output behavior.
- Float32 operational checks must assert actual leaf/storage dtypes; keep float64 reference owners.
- Runtime gains must come from removed work or reused compilations, with numerical work still executed.

## Task 1: Establish authoritative baseline and bottlenecks

- [x] Confirm all 161 source/harness hashes match the latest 1285-case full validation.
- [x] Inspect historical XML and static harness call paths; independently audit new lifecycle coverage.
- [x] Finish fresh full CPU baseline with coverage and GNU time resources; save test/source hashes.
- [x] Record the largest costs and coverage ownership before changing maintained tests.

## Task 2: Reuse identical setup and reference work

Files: `tests/test_restart.py`, `tests/lifecycle_sharding_probe.py`.

- [x] Build one module-scoped valid restart and enabled-output restart template; copy to isolated per-test paths before any mutation.
- [x] Build each serial reference once per preset/boundary and reuse across layouts, retaining all 20 real four-CPU runs and full-array/output checks.
- [x] Verify all corruption cases remain individually collected; inspect pristine template bytes after consumers and run affected tests.

## Task 3: Consolidate redundant contracts and reduce operational precision

Files: `tests/test_restart.py`, `tests/test_restart_contract.py`, `tests/test_simulation_output.py`, `tests/test_simulation_io.py`; additional changes require explicit audit evidence.

- [x] Fold diagnostics restoration and cyclic-halo roundtrip assertions into the existing all-field restart owner and remove redundant standalone trajectories.
- [x] Fold selected-load diagnostics/forcing identity assertions into the existing derived-field/time owner.
- [x] Replace duplicated checkpoint/dtype Cartesian rows with representative float32 operational cases and a small float64 cross-precision sample; keep every preset, boundary and checkpoint-position contract.
- [x] Remove repeated dtype-independent output sampling rows while keeping both initial-sampling branches and both storage dtypes.
- [x] Run affected correctness tests; compare all maintained executed lines with the baseline and inspect any differences before accepting.

## Task 4: Evaluate compilation overhead from evidence

Files: `tests/conftest.py` only if measurements justify a harness cache; tests for any new harness behavior precede implementation.

- [x] Inspect persistent compilation cache behavior in installed JAX, then verify reuse in a real repeated cold-process compilation.
- [x] Accept only a demonstrated safe cache or execution change; honor caller settings and isolate/clean resources.
- [x] Record unsuccessful approaches and avoid speculative changes.

## Task 5: Verify and document the complete result

Files: `tests/AUDIT.md`, `CHANGELOG.md`, this plan; performance evidence in `test_logs/test-runtime/`.

- [x] Obtain independent review of removals, retained assertions, precision and fixture isolation.
- [x] Run final full correctness with identical coverage/resources settings; compare per-file executed lines and case/module costs, explaining any precision differences.
- [x] Run maintained Ruff/format/annotations/ty, dependency check, strict Sphinx, fresh wheel/sdist and clean installed public examples.
- [x] Record measured wall time, peak process RSS, dtype storage savings, preserved coverage and hardware limits.
- [x] Commit the focused passing change on `jax-only` when all gates pass.

## Execution ledger

- 2026-10-11: Baseline root process is live (exec session 70431). Drafts stay in `test_logs/test-runtime/draft/` until baseline completes; production/tests used by the measurement remain unchanged.
- Ruling: work in the user-designated `jax-only` checkout and continue authorized investigation/optimization without another approval handoff. User and developer autonomy/workflow instructions take precedence over optional skill gates.
- 2026-10-11: Baseline passed 1283 tests with two GPU skips in 2670.82 s (external wall 2681.20 s; peak process RSS 3910968 KiB). Integrated independently reviewed drafts only after baseline completion. Cache regression RED precedes implementation; GREEN passes both tests with different numerical inputs and an actual executable-cache hit. Final full validation is running in exec11611; root retains sole pytest ownership.
- 2026-10-12: Final full validation passes 1273 tests with the same two GPU skips in 2210.64 s; external wall 2219.44 s is 17.22% lower. All maintained executed/missing/excluded line sets match (4058/4296); tested hashes and scientific fixtures match. Measured float32 State storage halves (22400 vs 44800 bytes, 70 fields), while whole-process peak RSS rises 2.69% to 4016260 KiB. Report this limit explicitly; module-boundary cache release remains unmeasured and is not introduced. Maintained static/dependency and strict Sphinx checks pass; package/installed verification and commit remain.
- 2026-10-12 delivery: fresh wheel/sdist and clean installed public imports, coupled exact restart, observations, selected input, custom ocean, two-CPU sharding and pure JIT/grad/JVP/VJP all pass. Wheel bytes match all 55 maintained production modules; new cache regression/audit/plan bytes match the source archive. All required gates pass; the focused change is ready for the prescribed jax-only commit. Test-only runtime improvements do not establish a whole-process memory reduction.
