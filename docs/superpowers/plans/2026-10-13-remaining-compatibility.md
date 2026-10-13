# Remaining Compatibility Removal Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; specialized audit,
> documentation and final review assistance follows AGENTS.md.

**Goal:** Finish the entire-source compatibility audit and remove valueless APIs.
**Architecture:** Each current domain owns its contracts; Experiment is the
single preset initialization interface. Scientific kernels and persisted schemas
retain their exact behavior. Native execution is already user-authorized.
**Tech Stack:** Python 3.14, JAX, pytest/xdist/coverage, Ruff, ty, Sphinx.
**Spec:** `docs/superpowers/specs/2026-10-13-whole-source-audit.md`

## Global Constraints

Work and commit on jax-only; one pytest session at a time; preserve scientific
kernels, reference bytes, tolerances and prescribed-ocean examples; full CPU
correctness <=1200 seconds and maintained coverage >=80%; native GPU validation.

## Review Focus

- Import order stays cold and no JAX configuration changes occur at import.
- All unequal-clock numerical references survive without lifecycle bypasses.
- Explicit NamedSharding and host-only mesh contexts preserve primal/AD probes.
- Profile/output defaults remain authoritative without parser default insertion.
- Persisted version-one schemas and geometry conversion stay intact.

### Task 1: Source and caller cleanup

Files: setups, model/experiment.py, io/cli.py, grid/halos.py, tests, benchmark.
Interfaces: Experiment.builtin/initialize_experiment; runtime mesh/bootstrap;
Click parse_options/driver_definition; fill_overlap with host mesh context.

- [x] Audit every source domain and record removals and retained value.
- [x] Extend ownership regressions; observe 8 failures before deleting APIs.
- [x] Remove setup translation/subclass, runtime reexports/parser, duplicate
  output parser/path helper and halo binding; migrate direct callers.
- [x] Observe 2 new module-runner validation-error failures, then remove error
  translation and keep rejection/no-output assertions.
- [x] Migrate scientific, numerical, AD, sharding and CLI tests; replace adapter
  self-comparisons with direct prescribed-field/shape/precision assertions.
- [x] Verify targeted contracts and resolve remaining migration failures.

### Task 2: Documentation, verification and commit

Files: active API/reference docs, hierarchy, DESIGN.md, CHANGELOG.md, audit.
Interfaces: canonical APIs from Task 1; installed package and public examples.

- [x] Migrate removed initialization/halo examples and refresh hierarchy.
- [x] Run Ruff lint/format/annotations, ty and shell syntax.
- [x] Run strict Sphinx and clean wheel/sdist/installed documented probes.
- [x] Run full CPU suite/coverage and native GPU contracts for the cleanup.
- [x] Obtain independent review, address material findings and verify.
- [x] Run the user-requested final two-node/two-rank aegir job and compare output.
- [x] Record exact accounting/scientific invariance; commit on jax-only.

### Task 3: Remove duplicated growth run controls

Files: setups/run_growth.py, setups/presets.py, tests/setups/test_presets.py;
obsolete test-only lazy import helper/fixture in dynamics/island tests.
Interfaces: GrowthSettings accepts only physical column scenario and dtype;
SetupPreset owns grid defaults; profiles own default run duration.

- [x] Observe three old scenario run-control constructors fail removal tests.
- [x] Move existing 2x2/150-step defaults to provider/profile ownership without
  changing values, remove duplicate dataclass/registry fields and simplify the
  scenario extraction now that no scenario includes run controls.
- [x] Remove unused test-only import adapters and retain every numerical check.
- [x] Verify focused controls and rerun full/static/docs/package/GPU gates on
  final source; review the small extension before committing.

### Task 4: Resolve the final two-node correctness failure

Files: execution/advance.py, distributed framework and closed-wall probes/tests,
audit records.
Interfaces: nested dynamics rollout and final field-aware halo refresh.

- [x] Reproduce intermittent two-node divergence and rule out settings, mesh,
  initialization, checkpointing and collection; optimized programs match ranks.
- [x] Compare twelve baseline and ordered trajectories: baseline fails twice,
  ordered control passes all twelve, checking all 70 physical fields per rank.
- [x] Run a maintained regression on aegir before changing production source.
- [x] Establish the physics-to-final-halo ordering without changing equations;
  native32-repeat regressions pass both1x2/2x1 and actual executor buffer graphs
  verify the completion dependency after lowering.
- [x] Restrict the barrier to multi-process execution after a broad local barrier
  exposes an in-process collective AD collision. Preserve the original local
  probe; extend it to controllers with explicit State inputs and host gathering.
- [x] Verify both closed-wall slip choices and all halo/JVP/VJP prelude checks
  on four native ranks; focused local/distributed derivative cases pass 2/2.
- [x] Review and verify the regression, CPU suite, AD/GPU and installed gates.
- [x] Finish with a fresh two-node aegir CLI run and serial equivalence.

## Ledger

Ruling: the existing user cleanup authorization and native execution apply;
no renewed stage approval is needed. Work remains in the specified checkout.
Ruling: retain halo-inclusive ocean conversion and version-one schema metadata
because they serve concrete current contracts. Remove only APIs with no such
value. Preflight: docs/tests consume Task 1 APIs; root owns source and pytest.
Evidence is recorded in CHANGELOG.md and test_logs/compatibility-audit/.
