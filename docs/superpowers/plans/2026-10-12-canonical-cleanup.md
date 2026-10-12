# Canonical Cleanup Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans with specialized
> source, test-quality and documentation assistance required by AGENTS.md.

**Goal:** Remove every module and wrapper retained solely for legacy imports.

**Architecture:** Canonical owners remain authoritative. Current lifecycle,
CLI and scientific preset factories consume those owners directly. Scientific
oracles and AD assertions remain independent of removed wrappers.

**Tech Stack:** Python 3.14, JAX, pytest/xdist/coverage, Ruff, ty and Sphinx.

**Spec:** `docs/superpowers/specs/2026-10-12-canonical-cleanup.md`

## Global Constraints

- Work and commit on `jax-only`; one coordinated pytest session at a time.
- Preserve numerical equations, reference bytes, dtypes and AD tolerances.
- Full CPU correctness in at most 1200 seconds; maintained coverage at least 80%.
- Verify both native GPU contracts and clean installed documented examples.

## Review Focus

- Fresh canonical import orders must neither enumerate devices nor alter JAX flags.
- Preset CLI execution must retain output, calendar and rectangular grid behavior.
- Frozen primals and finite differences must remain independent numerical oracles.
- Distribution must retain per-partition halos and process-zero output collection.
- Wheel/sdist must include canonical modules and no obsolete source scaffolding.

### Task 1: Canonical source and tests

Files: `veris/`, `tests/`, `benchmarks/`.
Interfaces: existing canonical `advance`, `step`, `run_timed`, geometry, arrays,
distributed collector and storage contracts replace obsolete paths/wrappers.

- [x] Write architecture regressions for canonical-only package ownership and
  absence of deprecated setup exports; observe failures before deletion.
- [x] Delete legacy module files and schema reexports; update canonical imports.
- [x] Delete setup timestep/output wrappers and fold dispatch into advance.
- [x] Remove alias tests; migrate scientific/gradient/CLI callers and benchmark.
- [x] Rehome unequal-time scientific reference under canonical kernel tests.
- [x] Run architecture development verification; omit a redundant fast run
  because the full correctness suite is planned, as directed by AGENTS.md.

### Task 2: Documentation and delivery

Files: README/DESIGN/CHANGELOG, active `docs/`, test audit docs, CI.
Interfaces: task1 canonical source paths and maintained CLI/lifecycle contracts.

- [x] Remove old shim trees/mappings/pickle claims and migration guide/links.
- [x] Replace setup wrapper examples with canonical execution and I/O.
- [x] Delete superseded compatibility spec/plan; remove obsolete lint exception.
- [x] Run required static checks, strict docs and clean package/install probes.
- [x] Run full cold CPU suite with coverage and both native GPU contracts.
- [x] Obtain independent whole-change review; resolve material findings.
- [x] Record exact test accounting, numerical invariance and verification.
- [x] Commit verified cleanup on `jax-only` after the completed gates.

## Ledger

Ruling: the user's explicit removal request supersedes earlier compatibility
requirements. No extra design approval is needed to carry out this cleanup.
Ruling: preset initialization and curated root/io APIs remain current features;
they are not legacy import shims. No numerical feature is added.
Pre-flight: task2 docs/installation consume task1 canonical names; root owns
source and coordinated tests, assistants edit disjoint tests/docs.

Verification: 1310 CPU passes, two expected GPU skips, and both native GPU
contracts pass. Cold CPU external wall is 756.48 seconds; maintained coverage
4056/4289 (94.57%), with no unchanged coverage losses. All 181 final Python-source
hashes and ten reference files are unchanged during verification. Clean packages
contain 65 Python modules and no obsolete modules; installed literal examples
and all required static/documentation gates pass. Whole-change review has no
remaining Critical/Important findings. Evidence: `test_logs/cleanup/`.

Failed approaches and fixes are recorded in CHANGELOG.md: Python 3.9 cannot parse
the maintained aliases; scheduled output needs the State selector from the
canonical carry; the first full run exposes 14 stale harness contracts. The
fresh full run verifies their corrections. The final suite has 83 fewer cases
than the 1395-case baseline; numerical/AD assertions and frozen archives remain.

User clarification: double only the four-CPU scientific subprocess timeout to
2400 seconds. There is no unit assertion of a 20-minute full-suite duration.
