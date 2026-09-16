# Test suite compaction implementation plan

> **For agentic workers:** Use Superpowers audit and verification workflows.
> Root schedules the only pytest instance; specialists edit disjoint test groups.

**Goal:** Reduce repeated test execution and test maintenance while retaining
independent physical oracles, branches, precision, AD and integration contracts.

**Architecture:** Audit assertions and inputs, not line overlap alone. Batch
pointwise cases, combine assertions on identical trajectories, and give each
contract an owning test. Production code stays unchanged.

**Tech Stack:** pytest, JAX, coverage.py, Ruff, ty.

**Spec:** User's active test-suite audit goal; DESIGN.md and AGENTS.md.

## Constraints

- Use existing `.venv-latest`, `jax-only` source and one pytest process at a time.
- Preserve independent equations, edge cases, CPU/GPU boundaries and AD checks.
- Do not lower tolerances, skip failures or add coverage exclusions.
- Full correctness and coverage comparison before committing.

## Tasks

- [x] Inspect baseline collection, source hashes, full-suite timings and GitHub
  commit. Baseline 1059 collected; 1057 passed and two GPU-only skips.
- [x] Audit physics/AD tests: remove duplicate parameter products, batch
  elementwise examples, reuse primal outputs from derivative evaluations.
- [x] Audit configuration/typing/initialization: consolidate schema assertions,
  checker invocations and repeated initialization into existing contract tests.
- [x] Audit output/storage: consolidate duplicate guards, snapshot and callback
  assertions while preserving calendar, lifecycle, distributed and memory cases.
- [x] Audit setup/rollout/integration: retain three-step real-physics comparisons,
  let recurrence tests own count boundaries, remove identical ocean AD reruns,
  and fold nested-directory checks into zero-step CLI coverage.
- [x] Document every removal and retained owner in tests/AUDIT.md; review all
  remaining test modules, including distributed probes and benchmark tests.
- [x] Run fast development suite, inspect any failure, run full CPU coverage,
  compare executed maintained lines with verified baseline, and run changed
  GPU tests where applicable. Run Ruff, formatting, annotations and ty.
- [x] Record exact collection/actions reductions and limitations in CHANGELOG;
  commit verified changes on jax-only.
