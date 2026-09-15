# Development log

## 2026-09-15 — Uninterrupted versus restarted integration

- [x] New full-State equivalence test uses coupled artificial
  dynamics, advection and growth on a nonuniform 6 by 8 island grid. Checkpoints
  after steps 1 and 3, float32/float64, and changing cooling exercise physical
  netCDF fields, configuration/constants, elapsed time and explicit halo rebuild.
- Fresh State arrays are poisoned before loading so successful continuation
  cannot rely on retained arrays. Compare all State fields exactly, and include
  an omitted-halo-rebuild negative control. All 25 focused CPU tests pass.
- Initial test construction exposed derived dataclass fields and JSON tuple
  conversion: restore constructor fields only and convert coefficient lists
  back to tuples. Nondefault rhoAir and exact time assertions prevent silent
  fallback to default constants or an incorrect forcing step.
- [x] New float32 case exposed a production metadata serialization failure.
  Dedicated storage test reproduced it; a narrow np.floating-to-float JSON
  handler fixes scalar and nested tuple metadata without changing numerics.
- [x] Independent review found no remaining blocker. All four restart cases
  also pass on CUDA (Tesla P100), with exact same-backend State comparisons
  and no skips. Full CPU suite: 879 passed, one existing GPU-only skip, zero
  failures/errors in 422.52 s. Maintained coverage 2341/2420 =
  96.74%; whole-package 84.42%. All maintained Ruff/format/annotation/ty
  gates pass. Source and test hashes match the full-suite inputs.
- Evidence: test_logs/restart-focused-green.log, restart-gpu-results.xml,
  restart-full-results.xml, restart-coverage.json and restart-source-hashes.json.
  Tests establish manual serial same-backend restart equivalence on CPU/CUDA;
  automatic distributed restart remains outside the existing input API.

## 2026-09-15 — Documentation and output policy migration

- [x] Implemented all five requested changes on jax-only.
  Docs agent owns Sphinx migration; CLI agent owns Click/netCDF drivers;
  averaging agent owns complete-window scheduling; root owns halo-free storage
  and serialized test execution.
- Moved doc/ content to docs/, preserving existing Superpowers records; updated
  CI, Read the Docs, source packaging and build instructions.
- Regression tests reproduced halo-inclusive snapshots, partial mean writes,
  old NPZ driver output and obsolete options. Storage/distributed focused checks
  now pass 21 tests after always trimming serial/partition halos. Snapshot input
  inserts physical interiors while preserving initialized halos for exchange.
- Environment: established .venv-latest is active; .venv does not exist.
- [x] Click replaces argparse in drivers, benchmark and reduction probe. Final
  driver output defaults to netCDF; --final-netcdf aliases --output. No NPZ
  writer remains (the immutable numerical-reference NPZ input is retained).
- [x] Removed include_halos/write_partial configuration and API arguments.
  All 34 averaging tests pass; incomplete initial/final windows are discarded.
- [x] Driver/integration/benchmark checks pass 39 tests with one GPU-only skip.
  Independent review caught bypassed parallel finite validation; a new main-path
  regression reproduced it, then passed after restoring physical-field checks.
- [x] Sphinx -E -W build passes with network inventory access. All 18 original
  tracked source files have docs/ destinations; rendered registry tables omit
  removed options. Ruff, formatting, annotations and ty checks pass.
- [x] Full CPU suite: 874 passed, one existing GPU-only skip, zero failures/errors
  in 407.52 s, including the two-process reduction/AD tests. Maintained coverage
  2338/2416 = 96.77%; whole-package coverage 84.43%. No additional exclusions.
- [x] Follow-up GPU validation: GPU-only parallel CLI/netCDF test passed (1/1,
  zero skips) on two Tesla P100 GPUs. Output fields are finite and have the
  physical 12 by 16 shape. Initial sandbox CUDA access failed before testing;
  rerun with device access succeeded. Evidence: test_logs/migration-gpu-test.log
  and migration-gpu-results.xml. Production sources unchanged.
- [x] Final independent review verified the parallel finite-check repair and all
  five source requirements. Source archive includes docs/ and no doc/. Production
  and test hashes match the full-suite snapshot; no pytest process remains.
  Evidence: test_logs/migration-{full-tests.log,results.xml,coverage.json,
  source-hashes.json,docs.log}, migration-docs/ and migration-dist/.

## 2026-09-15 — Calendar-aware I/O design

- [x] Implemented veris.io calendars, immutable registry-backed output options,
  per-stream exact sampling, float64 sum/count means, h5netcdf storage/input and
  independent final snapshots. State and numerical kernels stay unchanged.
- [x] Added opt-in netCDF controls to growth/dynamics/parallel drivers; collective
  output removes every partition halo and writes only on rank zero. Retained NPZ.
- [x] Focused checks cover Gregorian/fixed dates, daily/monthly/annual/custom
  means, subsecond/multiple schedules, constant buffer size, snapshots/input,
  grad/JVP/JIT no-effects (including constants), actual growth AD final State,
  and four-local-CPU physical output. Baseline fast suite passed 79 tests.
- [x] Independent review reproduced silent dtype loss and a rejected sample
  contaminating an earlier mean stream. Added regressions, reject unsafe storage
  casts and preflight all stream schemas before updating accumulators.
  Another regression corrected discard-final-partial behavior to retain a
  completed initial partial period. No physics equations or tolerances changed.
- [x] Ruff/annotations/ty pass; Sphinx clean warnings-as-errors build passes.
  Initial docs attempt found a short heading underline and sandbox inventory
  access failure; repaired heading and rebuilt with network access.
- [x] Full CPU correctness: 868 passed, 1 existing GPU-only test skipped,
  0 failures/errors. SLURM job 65265147 COMPLETED exit 0:0 in 6m51s on node453,
  partition aegir with constraint v3, exactly as requested.
- [x] Coverage: maintained 2312/2415 = 95.73%; whole package 2312/2768 =
  83.53%; new I/O package 495/530 = 93.40%. No omissions beyond the existing
  generated _version.py exclusion for the maintained metric.
- [x] True two-process CPU output matches the physical NPZ fields exactly;
  instantaneous/final timestamps, three-sample daily mean metadata and 360_day
  February 30 verified. Both ranks participated; one netCDF writer succeeded.
  CPU validation used the requested aegir/v3 allocation; no GPU partition used.
- [x] Sphinx clean build and rendered registry-table inspection pass. Independent
  re-review found no remaining blocker; tested production/source hashes match.
  Evidence: test_logs/io-validation-65265147/, test_logs/io-docs.log and
  test_logs/io-source-hashes.json. No pytest process remains in the completed job.
- [x] Requirement audit recorded in docs/superpowers/plans/2026-09-15-io.md;
  verified implementation ready for local integration back to jax-only.


- User explicitly approved design; implementation resumed. Root owns storage,
  scheduling, drivers and pytest scheduling; calendar specialist owns calendar
  module/tests. Plan: docs/superpowers/plans/2026-09-15-io.md.

- Inspected authoritative jax-only tree: no existing I/O/calendar subsystem;
  maintained examples currently write NPZ. VARIABLES supplies netCDF metadata,
  and the parallel runner already provides partition-aware halo removal/gather.
- Proposed complete I/O design in
  docs/superpowers/specs/2026-09-15-io-design.md, including input, calendar
  sampling, sum/count means, distributed output and strict AD separation.
- Awaiting design approval required by the explicitly invoked Superpowers
  brainstorming skill. No implementation or test run yet.
- Blocked audit: approval remains absent across the initial design turn and two
  automatic continuations. Previous turn made no implementation progress;
  rechecked worktree and proposed spec. Goal marked blocked pending explicit
  design approval; full implementation scope remains unchanged.

## 2026-09-14 — Shared Parameter metadata

- Replaced Setting and PhysicalConstant with one Parameter named tuple in
  _typing.py, retaining default, type, description and optional units fields.
  Both main registries, setup registries, metadata helpers and public settings
  exports now use Parameter. Updated ownership tests and current documentation.
- The new shared-registry regression failed before implementation; all 51
  focused ownership/configuration tests pass. Independent review found no
  blockers and confirmed all 219 registry entries preserve their keys, defaults,
  types, descriptions, units and ordering against HEAD.
- Broader tuple-capable default typing exposed one scalar setup read; added an
  explicit float cast. Maintained Ruff, formatting, annotation and ty checks pass.
- Warnings-as-errors Sphinx build passes after a network-enabled retry for the
  intersphinx inventory. Initial full CPU run completed with only the sandbox
  denying a local socket in the two-process reduction test.
- [x] Unrestricted full CPU suite: 785 passed, 1 skipped in 370.34 s.
  Maintained coverage: 1755/1860 statements (94.35%), passing the 80% gate.
  Logs: test_logs/parameter-full-cpu.log and parameter-coverage.json.
  Completed on jax-only for the requested commit and push; local test artifacts
  remain untracked.

## 2026-09-14 — Review settings and physical-constant units

- Reviewed all 34 SETTINGS and 145 PHYSICALCONSTANTS entries. AST comparison
  against HEAD confirms identical keys, numerical defaults and declared types;
  independent review confirms the dataclass schemas and runtime code are unchanged.
- Completed the pending separation of units from descriptions. Corrected units
  for linear ocean drag, strain-rate regularization, Dalton transfer, lead-closing
  thickness and reciprocals, basal drag, coastal drag, and the longwave pressure
  safeguard using their kernel equations. Clarified cpvir, CrMax and the unused
  legacy explicitDrag flag. No numerical formulas or tolerances changed.
- Maintained Ruff, formatting, annotation and ty checks pass. Sphinx builds with
  warnings as errors. The first sandbox build could not fetch an intersphinx
  inventory; its attempted dictionary CLI override was unsupported. The normal
  network-enabled rebuild passed.
- [x] Full CPU suite: 708/708 passed in 288.38 s; maintained coverage
  1441/1452 (99.24%) passes the 80% gate. Logs: test_logs/registry-units-full-cpu.log
  and registry-units-coverage.json. Final independent review found no blockers.
  Prepared for the requested commit and push to origin/jax-only; local
  test/profiling artifacts remain untracked.

## 2026-09-11 — Remaining physical coefficient ownership

- Audited all 15 remaining float-valued SETTINGS entries and kernel consumers.
  User clarified that numerical controls and all timesteps remain Configuration,
  and explicitly kept nITC/recip_nITC together. Adapter initial temperature also
  remains a configuration input. Independent semantic review identified
  pressReplFac as the remaining constitutive-law coefficient.
- Moved pressReplFac with its unchanged 1.0 default to PhysicalConstant metadata
  and the frozen PhysicalConstants class. Both pressure-law reads now use phys.
  Metadata-driven documentation and the ownership inventory reflect the change.
- Ownership regression failed before production changes. Scalar pressure tests
  now include the fractional weight 0.5, preserving continuous float behavior.
  Added initialization/precision/validation checks and explicit numerical-control
  retention assertions. No new physical bounds or numerical tolerances imposed.
- Rejected broader migration after user clarification: EVP relaxation controls,
  clipping and sqrt safeguards are numerical parameters; float type alone is
  not an ownership rule. Temporary broader test/docs edits were reverted.
- Preliminary narrow migration: 86 focused and 71 fast CPU tests passed;
  maintained Ruff, formatting and ty passed. Initial .venv activation failed
  because this checkout uses the established .venv-latest environment.
- Maintained static checks and warnings-as-errors Sphinx build pass. The first
  Sphinx attempt could not fetch intersphinx inside the sandbox; the approved
  network-enabled rebuild passed.
- The user interruption terminated the first full CPU run at 81% without a final
  result. Revalidated missing process handle and absence of pytest before
  restarting the full suite on final files, including the dtype assertion fix.
- [x] Final full CPU suite: 708/708 passed in 212.06 s. Maintained coverage
  1441/1452 (99.24%) passes the 80% gate. Evidence is in
  test_logs/pressure-ownership-full-cpu.log and pressure-ownership-coverage.json.
- [x] Final independent review found no blockers and confirmed all numerical
  controls remain in Configuration. Ruff/format/annotations/ty, Sphinx and
  git diff --check pass. No tests remain live; no production changes after
  final-suite validation. Ready to commit on jax-only; no remote push.

## 2026-09-11 — Fresh external ocean initialization and setups package

- Moved precision metadata directly into SETTINGS["dtype"] and removed PRECISION.
  Dependent configuration defaults and validation read the dictionary entry.
- Renamed setup to setups and migrated active imports, tests, benchmarks and docs.
- Replaced set_inits with setups/ocean.py:initialize_from_ocean. It infers interior
  dimensions from halo-inclusive ocean geometry, computes the original staggered
  metrics, and supplies them to the shared initialize for fresh State allocation.
  Optional settings, physical and state overrides initialize coupling inputs.
- Migrated numerical geometry tests before implementation and observed the missing
  new-package import failure. All 44 focused cases pass, including float32/64,
  fresh defaults, external forcing and invalid overrides. Independent review
  found no blocking issues. Maintained Ruff, formatting, annotations and ty pass.
- Warnings-as-errors Sphinx build passes after enabling network access for its
  intersphinx inventory. The initial full CPU run had only the sandbox-denied
  local socket failure in the two-process reduction test.
- Full unrestricted CPU suite passes 699/699. Maintained coverage is
  1441/1452 (99.24%), passing the 80% gate. Final source passes Ruff, formatting,
  annotation checks, ty, Sphinx and git diff --check. Evidence is retained in
  test_logs/ocean-full-cpu.log, ocean-coverage.json and ocean-docs.log.
- Completed on jax-only; no remote push. Generated test artifacts stay untracked.

## 2026-09-10 — Configuration class and conf arguments

- Renamed Settings to Configuration in configuration.py and its public re-export,
  imports, annotations, initialization, tests, documentation and benchmarks.
- Renamed every old object identifier sett to conf, including keyword calls,
  pytest fixtures and JAX static_argnames. SETTINGS and Setting metadata remain.
- Updated tests first and observed the expected missing-Configuration import
  failure. Independent review found no correctness issues; grammar notes fixed.
- Ruff, formatting, annotation checks, ty and warnings-as-errors Sphinx build pass.
  The initial documentation build required network access for intersphinx.
- Initial full CPU run had only the sandbox-denied local socket failure in the
  two-process reduction test. The unrestricted full CPU suite passes 693/693.
  Maintained coverage passes the 80% gate. Evidence is in
  test_logs/configuration-rename-full-cpu.log and configuration-rename-coverage.json.
- Final identifier audit and git diff --check pass. Committed on jax-only;
  no remote push performed.

## 2026-09-10 — explicit metadata owners and OceanGeometry

- Applied the user's precise type locations: Setting/configuration.py,
  PhysicalConstant/physical_constants.py, Variable/variables.py, and
  Diagnostics/diagnostics.py. Renamed the central Geometry to OceanGeometry
  throughout callers and tests, without a stale alias.
- PRECISION is now a single Setting entry in configuration.py, referenced by
  SETTINGS["dtype"]. Removed the Precision base class; each configuration
  dataclass has an explicit keyword-only dtype field with the same eager default.
- Added expected-red ownership/name/import tests before moving definitions.
  The interrupted commit never executed; prior 688-case CPU/GPU results
  precede these latest corrections and are not final-source evidence.
- [x] All 93 focused regression cases pass, followed by the full CPU suite:
  693/693 in 211.50 s. Maintained coverage 1431/1442 (99.24%); the established
  80% gate passes. Whole-package coverage is 79.72% including generated code.
  Evidence: test_logs/explicit-owners-full-cpu.log and explicit-owners-coverage.json.
- [x] Ruff/format/annotation/ty and warnings-as-errors Sphinx HTML build pass.
  Independent review found no runtime issues; corrected its documentation notes.
  Final ownership audit verifies every requested module and removed old names.
- [x] All 69 targeted GPU precision/state/diagnostics/geometry/initialization
  cases pass with an asserted GPU backend. Log: test_logs/explicit-owners-gpu.log.
  No source changes after CPU validation and no pytest remains running.
- Ready to commit the complete corrected refactor on jax-only. No remote push;
  large test logs and generated artifacts remain untracked.

## 2026-09-10 — configuration ownership and central type definitions

- [x] Moved EVP stress/shear coefficients to PhysicalConstants after tracing
  their influence on the converged stress law. Defaults and positive validation
  are preserved; timestep/relaxation-rate/tolerance controls remain Settings.
- [x] Moved all 12 artificial scenario controls and validation into artificial.py.
  scenario_overrides configures initialization; step(cooling=...) supplies custom
  cooling. Generic Settings and State contain no artificial-only controls.
- [x] Centralized shared model/metadata/geometry/diagnostics types in _typing.py.
  Per the user's explicit correction, Settings stays with SETTINGS in
  configuration.py and PhysicalConstants stays with PHYSICALCONSTANTS in
  physical_constants.py. ArtificialSettings remains local to the setup.
  Removed state.py, _bulk_types.py, _solver_types.py and _thermodynamic_types.py;
  migrated consumers and documentation.
- Rejected approach: putting Settings/PhysicalConstants in _typing.py exceeded
  the requested ownership boundary and required unnecessary lazy defaults.
  Restored eager registry defaults and removed the deferred machinery/tests.
  Added expected-red tests explicitly enforcing the two local class exceptions.
- The first 123 focused cases, static checks and docs passed before this
  correction. Interrupted the obsolete full CPU run at 359 passing tests;
  those results are not final-source validation. No numerical tolerances changed.
- Final consumer audit found saltOcn_ref was used only by the artificial setup;
  moved its unchanged 34.7 default to ARTIFICIAL_SETTINGS and tested scenario
  overrides. Interrupted the second obsolete CPU run at 326 passing tests.
  An AST audit of every registry key found no other artificial-only consumers.
- [x] Corrected final-source CPU suite passes 688/688 in 209.86 s, including
  distributed collectives. Maintained coverage 1427/1438 (99.24%); whole package
  79.68%. The maintained 80% coverage gate passes. Log and coverage JSON:
  test_logs/ownership-full-cpu.log and test_logs/ownership-coverage.json.
- [x] Ruff/format/annotation/ty gates and Sphinx HTML with warnings as errors
  pass. Updated independent review found no correctness/import/default issue;
  replaced a redundant settings identity assertion with public-reexport coverage.
- [x] Final asserted GPU-backend suite passes 688/688; CPU-only integration
  checks retain their intentional backend. Log: test_logs/ownership-full-gpu.log.
  Final source is unchanged since CPU validation; no pytest remains live.
- [x] Completion audit verifies local Settings/PhysicalConstants classes,
  disjoint scenario ownership, centralized remaining shared types, removed
  redundant modules and generated registry documentation. Ready to commit on
  jax-only; test logs and generated artifacts remain local. No remote push.

## 2026-09-10 — initialization dtype policy

- [x] Defined the shared frozen keyword-only dtype field once in PRECISION;
  Settings and PhysicalConstants inherit it. Initialization selects float32 or
  float64 for all 70 State fields, floating settings/constants, derived values
  and lookup tables. Integer and Boolean controls remain host static values.
- [x] Removed per-variable float64 policy. NetCDF examples use allocated array
  dtypes; growth/advection/EVP scratch arrays and artificial masks inherit State
  precision. Geometry conversion already followed State precision.
- [x] Added expected-red precision selection tests, then verified 42 focused
  cases and 66 fast regression cases. Review reproduced silent float32 derived
  ratio underflow; a failing regression now passes after computing dependencies
  from rounded inputs in host precision before their checked final cast.
- [x] Expanded checks to overrides, mesh placement, adaptive EVP, free drift,
  CESM stable/unstable fluxes and geometry conversion. Fixed a test fixture that
  accidentally promoted humidity inputs through a NumPy float64 intermediate;
  every bulk-flux input now has an explicit dtype assertion.
- [x] Maintained Ruff, format, annotation and ty checks pass. Sphinx HTML builds
  with warnings as errors and remote inventories disabled. Dependencies pass.
- First full CPU run found 16 failures: 15 float32 basal coefficient fixtures
  still used default float64 static constants; one schema test omitted inherited
  PRECISION. Updated fixtures to initialize matching precision, retained separate
  float64 analytic references and unchanged tolerances. All 72 focused basal and
  State tests pass. No production fix or numerical tolerance change was needed.
- [x] Final full CPU suite passes 673/673 in 208.11 s, including distributed
  collectives and initialized sharding. Maintained coverage 1433/1448 (98.96%);
  whole-package coverage 79.57%. Established maintained 80% gate passes.
  Logs: `test_logs/precision-final-cpu.log`, `precision-final-coverage.json`.
- [x] Independent final review confirms the fixture/schema changes preserve
  all numerical tolerances and analytic reference checks; no blockers remain.
- [x] Unsandboxed GPU-default full suite passes 673/673 in 345.01 s on the
  available P100 GPUs, with asserted GPU backend and CPU support retained for
  callbacks/CPU-only tests. Log: `test_logs/precision-final-gpu.log`.
- [x] Final completion audit: one precision default; complete scalar/derived/table
  normalization; all State and scratch allocation paths propagate precision;
  physical ownership/legacy-default oracles pass; no config fields enter State;
  output docs use actual array dtype. Both full suites and maintained checks pass.
  Multi-device integration retains its existing float64 oracle; both precisions
  additionally cover explicit mesh placement and coupled/alternative kernels.
- Ready to commit on `jax-only`; no remote push requested. Large test logs and
  coverage artifacts remain local and untracked. No test process remains live.

## 2026-09-10 — physical thresholds separated from execution settings

- [x] Moved 21 physical bounds, drag/viscosity regularization scales, wind floors
  and bulk-flux reference heights from SETTINGS/Settings to
  PHYSICALCONSTANTS/PhysicalConstants. Defaults and validation are preserved.
  This includes `hIce_min`, `basalDragMinArea`, `Area_reg`, `deltaMin`, `zref`
  and `bulkStabilityLimit`; consumers now read these through `phys`.
- [x] Added an ownership regression before implementation; root confirmed the
  expected missing-physical-registry assertion failure. Updated existing
  physical-law, nonsmooth-gradient and invalid-input tests to override physical
  constants, and updated generated-reference prose and coefficient inventory.
- Classification: physical thresholds define the constitutive/thermodynamic
  closure even with converged solvers. Grid sizes, boundary/pressure formulation
  switches, iteration counts, timesteps, EVP relaxation controls, `CrMax` and
  `eps2` remain numerical or execution configuration. Artificial forcing and
  initial conditions remain experiment configuration.
- Migration-only static type audit found no errors; integrated correctness and
  dtype validation are running in the root agent's single pytest lane.
- Environment note: `.venv` is absent; the existing `.venv-latest` environment
  supplies Ruff and ty. No environment was recreated.

## 2026-09-09 — simplify registry defaults and kernel annotations

- [x] Implemented registry-default and zero-default cleanup on `jax-only`.
  @simplify_types replaced State/Settings protocol variants with concrete types,
  removed two obsolete modules, and updated static contract tests.
- Added tests first for registry-populated constructor defaults, derived fields,
  frozen replacement and schema drift; confirmed three expected missing-helper
  failures before implementation. A shared `registry_defaults` decorator now
  fills Settings/PhysicalConstants defaults before standard dataclass creation.
  `FROM_REGISTRY` preserves optional typed constructor parameters without repeated
  key lookups/casts. Existing validation and derived computations are retained.
- Removed redundant zero defaults from VARIABLES; explicit nonzero defaults are
  retained. Updated initialization/interface documentation.
- [x] 49 configuration/initialization tests and 39 focused typing tests pass.
  Maintained Ruff, formatting, annotation coverage and ty checks pass; Sphinx
  documentation builds with warnings as errors and remote inventories disabled.
- [x] Independent review confirmed registry values, field order, validation and
  numerical bodies are preserved. Review caught runtime annotation resolution
  failing with TYPE_CHECKING-only imports; switched to ordinary concrete imports
  and strengthened the regression to call get_type_hints without injected names.
  Direct regression failed first and now passes all twelve kernel modules.
- Initial full CPU run: 645 tests pass; only two-process reduction fails because
  the sandbox denies socket creation. That run overlapped the annotation-import
  revision, so its coverage line mapping is stale and is not final evidence.
- [x] Final-source full CPU suite outside sandbox: 646/646 pass, including
  two-process collectives, gradients and concrete annotation introspection.
  Maintained coverage 1411/1426 (98.95%); whole package 1411/1779 (79.31%).
  The established 80% maintained-code gate passes. Logs and JSON coverage:
  `test_logs/simplify-final-cpu.log`, `test_logs/simplify-final-coverage.json`.
- [x] Final maintained Ruff/format/annotation/ty checks and Sphinx build pass.
  Independent fresh-process review confirms all thirteen affected modules
  import without cycles and resolve annotations. No physics, AD formulas or
  numerical tolerances changed. No remote push requested.

## 2026-09-09 — registry/dataclass initialization migration

- [x] User approved the design; working on `jax-only`. Instructions and design
  committed in c39447e. Implementation plan and coefficient inventory are under
  `docs/superpowers/`; no remote push requested.
- [x] Separate frozen Settings and PhysicalConstants are initialized from
  namedtuple metadata registries. All inventoried physical law coefficients,
  cloud tables, numerical controls and artificial experiment defaults are
  centralized. Exact derived values recompute on immutable replacement;
  independently rounded legacy defaults are preserved and checked against a
  historical 131-value JSON oracle from c39447e.
- [x] State is a frozen, registered 70-array JAX PyTree allocated from VARIABLES.
  Removed 14 unused/output-only fields. Five coupling outputs are returned in
  separate Diagnostics with output metadata. No configuration or mesh is in AD
  State. h5netcdf round-trip tests verify usable dimensions and attributes.
- [x] Migrated all maintained physics, structural protocols, geometry adapter,
  artificial integration, tests and benchmark callers to separate constants.
  Removed legacy combined Settings and mutable import-time halo configuration.
  Defaults, reference arrays and numerical tolerances remain unchanged.
- [x] Serial initialization validates grid extents, overrides and x64 precision.
  Explicit mesh initialization allocates every field in packed local-halo layout
  with NamedSharding; mesh resources remain outside the numerical State.
- [x] Four-device CPU initialized integration passes all 70 field comparisons,
  coupled stepping, cooling JVP/VJP and finite-difference comparison against a
  serial experiment (`initialized-sharding-third.log`). Global explicit-sharding
  roll failed in the first attempt: the driver now maps the entire stencil
  sequence onto local partitions and performs halo exchange in the manual mesh.
  A subsequent set/tuple comparison error was fixed before the passing run.
- [x] Targeted migration checks passed: dynamics/configuration/halo 220 tests;
  coupled diagnostics/oracles/benchmark 42 tests; geometry/configuration 45
  tests. These precede the final experiment registry and mesh-driver additions;
  the full suite below is the authoritative final-source validation.
- [x] Registry metadata review corrected wind staggering, reciprocal metric
  units, stress descriptions and reciprocal thickness description. Preserved
  the historical salt-flux equation; nonzero ice-salinity normalization remains
  ambiguous and its diagnostic units are explicitly documented as unknown.
- Failed migration approaches: a 59-test fast sample missed old artificial
  callers; focused integration found them. Reciprocal override tests now change
  independent base values; zero Area_reg remains permitted for existing oracle
  cases. Halo serial oracle inputs must have serial sharding, not partitioned
  explicit-sharding annotations. No physics tolerances were relaxed.
- [x] Maintained Ruff, formatting, annotation checks and dependency consistency
  pass; maintained ty check also passes. Final Sphinx build passes with remote
  inventories disabled through Python configuration; all 68 settings, 122
  physical constants, 70 state fields and five diagnostics appear in HTML.
  The first CLI override was ignored by Sphinx and attempted a remote inventory;
  the Python configuration avoids that unintended network dependency.
- [x] Full final CPU suite: 625/625 pass; maintained statement coverage
  1420/1443 (98.41%), whole package 1420/1796 (79.06%). The established CI gate
  excludes generated version metadata and passes. Full log and JSON coverage:
  `test_logs/dataclass-final-cpu.log`, `test_logs/dataclass-coverage.json`.
- [x] Final test-quality review added six invalid-mesh cases and compiled
  four-CPU step comparison. Follow-up initialization/distributed run passes
  19/19, including all 70 fields and halos; final mesh log retained. Independent
  code review found no blockers and confirmed registry/schema/field-use parity.
- First GPU launch used JAX_PLATFORMS=cuda, disabling CPU entirely: 622 pass,
  nine fail because residual debug callbacks and the explicit CPU benchmark
  require a CPU backend. This is a launch-configuration error, not a numerical
  discrepancy; no source or tolerance changes were made. Retained failed log.
- [x] Final full GPU-default suite with JAX_PLATFORMS=cuda,cpu: 631/631 pass
  in 305.54 seconds. Verified two CUDA devices and CPU callback availability.
  This run includes the six final validation cases and compiled distributed
  probe. Log: `test_logs/dataclass-final-gpu-cpu-enabled.log`. No pytest remains
  live. All required validation passes; reviewed migration is ready to commit
  on `jax-only`, with local test artifacts preserved and no remote push.
- Remaining known limitation: physical units of the historical nonzero
  ice-salinity coupling expression need separate scientific review; this
  refactor preserves that equation and documents the ambiguity explicitly.

## 2026-09-08 — initial test harness

- [x] Inspected `jax-only`; no existing tests, DESIGN.md, or CHANGELOG.md.
- [x] Read standalone `jax_halo_exchange` mesh initialization interface.
- IN PROGRESS (@root): environment, pytest fixtures, area/mass, averaging,
  cleanup, and free-drift numerical tests; next expand to other physics modules.
- [x] Test-quality agent reviewed transport, dynamics, and growth test cases.
- Environment attempt: uv default cache is outside writable sandbox; retry with
  escalation succeeded. `uv/latest` is the available module
  (`uv/lates` in AGENTS.md is a typo).
- Initial test status was unmeasured; verified results follow below.
- Discovered: halo exchange requires external `initialize_mesh_sharding`.
  Intensive advection references missing transports and potentially swapped axes;
  EVP residual mode has undefined symbols. Investigate with reproducing tests.
- Follow-ups: growth energy area factors and basal-drag float32 overflow deserve
  focused verification; do not treat source parity as proof of physical validity.

### Initial verified baseline

- [x] Created design, fixtures, stable collection-time `--fast`, and 55 tests.
- [x] Full suite: **55/55 passed**, 8.99 s. Area/mass 12/12, averaging 12/12,
  cleanup/ridging 16/16, free drift 9/9, mass gradients 6/6.
- [x] Fast selection: 6 passed, 49 deselected (before formatting).
- Coverage: **83/1520 statements, 5.46% overall** (rounded 5% report);
  area_mass, averaging, clean_up, and freedrift_solver each 100% statements.
- [x] New tests pass Ruff, formatting, and ty. Whole-repository checks report
  203 Ruff errors and 66 ty diagnostics; broad legacy cleanup not attempted.
- [x] Environment installed with approval; Python 3.9/JAX 0.4.30 selected by
  system interpreter. Pinned initial test environment in requirements-test.txt.
- [x] Added CPU CI configuration with measured 5% coverage floor. Remote CI
  has not run; increase floor toward 80% as physics coverage grows.
- NEXT: newer Python/JAX for source `jax.shard_map`, explicit standalone mesh
  integration, halo/transport tests, dynamics, growth/temperature, additional
  gradient checks, then close measured coverage gaps. Goal remains incomplete.

- Test-quality review found no numerical assertion defects in the baseline.
  Follow-ups: cache fixture namedtuple classes to reduce retracing; add positive
  area preservation, nonzero ocean drift, explicit uniform masks, and shape checks.

## 2026-09-08 — latest software and thermal balance tests

- [x] User requires latest available stable software. Installed Python 3.14.7,
  JAX/jaxlib 0.11.1, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1 and latest resolved
  development dependencies in `.venv-latest`. Activate this environment for
  subsequent work; old `.venv` remains untouched but is superseded.
- [x] Updated requirements-test.txt and CI to Python 3.14; raised coverage floor
  to 10%. Remote CI and GPU are not verified (local JAX lists one CPU device).
- [x] Added constructed conductive/radiative surface-energy equilibrium,
  absent-ice flux, and shortwave/snow-opacity cases for solve4temp.
- [x] Review improvements: explicit uniform masks, positive area preservation,
  ocean-driven free-drift momentum balance, shape assertions, cached fixture
  PyTree classes to avoid unnecessary recompilation.
- [x] Full suite on latest stack: **101/101 passed in 4.28 s**; area/mass 12/12,
  averaging 16/16, cleanup 17/17, free drift 27/27, gradients 6/6,
  surface temperature 23/23. Fast mode: 11 passed, 90 deselected.
- [x] Ruff check, Ruff format check, and ty pass for tests.
- Coverage: **154/1521 statements (10.12%)** across the entire package;
  solve4temp joins four previously tested physics modules at 100% statements.
  Statement-count change reflects the newer coverage/Python analyzer.
- Still next: actual halo/transport integration and conservation, dynamic
  rheology/EVP, growth energy budgets, broader gradients, legacy modules.
  Numerical coverage remains incomplete; 100% statements is not full physics
  validation. No production physics changed in this unit.

## 2026-09-08 — transport and physical budget expansion (in progress)

- IN PROGRESS (@root): serial halo import regression, periodic transport,
  ocean stress rotation, and execution/review of new numerical tests.
- IN PROGRESS (@test_quality): completed 47 dynamic rheology test cases;
  root full execution pending. Scalar equations and affine-field oracles.
- IN PROGRESS (@growth_tests): thermodynamic budget cases and explicit
  reproductions of suspected repeated-area factors, without xfail exemptions.
- Serial halo red/green: `--fast` first failed with missing external mesh
  initializer; moving its import into the sharded factory makes serial mode
  usable without that helper. 1 passed, 2 deselected after the fix.
- Intermediate fast suite: 18 passed, 155 deselected. Full suite and coverage
  pending final growth tests. Keep only one pytest process running at a time.

### Verified physics corrections and coverage

- [x] Growth reproductions: **6 failed, 27 passed** on original equations.
  Each category-reduced flux already represents a grid-cell mean; multiplying
  by concentration again breaks partial-cover budgets. Removed duplicate
  concentration factors in snow melt, Qsw, and net ocean energy.
- [x] Reproductions then **6/6 passed**. Independent review confirmed units
  and unchanged full-cover/ice-free limiting equations.
- [x] Added complete snow depletion/excess ice melting energy closure: **4/4**.
- [x] Real one-device shard_map halo test passes. Initial test input failed JAX
  0.11.1's explicit sharding check; device_put with NamedSharding supplies the
  actual required mesh placement. No production workaround was needed.
- [x] Full suite: **222/222 passed in 14.15 s**, coverage **452/1520 (29.74%)**.
  Tests pass Ruff, formatting, and ty. Counts include halo 4, transport 22,
  dynamics 47, growth 37, ocean stress 8 and expanded gradients.
- IN PROGRESS: wind/hydrostatic forcing tests (@test_quality); uniform EVP
  momentum checks (@root). EVP fast sample 1 passed, 6 deselected.
- Remaining risks: intensive transport, EVP adaptive/residual branches,
  nonuniform coastal gradients, multi-device/GPU execution, legacy Veros
  modules and setup; all remain explicit pending work.

### Final check for this development unit

- [x] **259/259 tests passed in 16.99 s** on Python 3.14.7/JAX 0.11.1.
  Wind/dynamic forcing 30/30; EVP uniform limits 7/7. Ruff, formatting,
  and ty pass for all tests. Full report: test_logs/coverage.json.
- Coverage now **577/1520 statements (37.96%)** across the whole package,
  including generated metadata and legacy setup. CI floor raised to 37%.
- Existing tests and all new physical-budget regressions pass without skips or
  xfails. Goal remains incomplete: target >=80%, further gradient validation,
  known branch failures, legacy dependencies, and actual distributed/GPU gates.

## 2026-09-08 — intensive transport and EVP diagnostics

- [x] Intensive transport regressions first failed 6/6 on missing `vs.uTrans`.
  Using the already computed local transports exposed 6/6 numerical failures
  from swapped divergence axes. Correcting axes passed all six. A seventh
  cross-flow test exposed use of the pre-zonal tracer in the meridional sweep;
  that compensation now uses the field entering the meridional sweep.
- [x] Adaptive EVP scalar momentum cases passed 8/8 without production changes.
  Residual cases failed 4/4 on undefined sigma11, then 4/4 on legacy update.
  Reconstruct previous physical stresses from principal components, use native
  JAX indexed updates, and exclude the actual fixed two-cell halo width.
- [x] Residual printing failed 2/2 with tracer formatting TypeError; replaced
  host formatting with jax.debug.print. Added direct numerical norm checks,
  including nonzero initial stresses, plus diagnostic-on/off solution parity.
- [x] Intermediate transport/EVP suite: 50/50 passed before two extra nonzero
  stress cases. Independent review found no must-fix defects in either fix.
- IN PROGRESS (@root): final full correctness/coverage run and checks.
- Limits: intensive tests currently use unit thickness; multi-device residual
  reductions are unverified and global_sum remains an identity. No GPU available
  in the checked local environment. These are not claimed as completed.

- [x] Final full suite: **282/282 passed in 23.91 s**; fast selection: 29
  passed, 253 deselected. All test lint, formatting, and ty checks pass.
- Coverage **607/1517 statements (40.01%)**, including generated and legacy
  files. CI floor increased to 40%. Statement count fell when obsolete residual
  update/unused ratio statements were removed. EVP reaches 100% statements,
  but the distributed and nonuniform-strain limits above remain unverified.
- Next: meaningful legacy heat-flux/setup tests, basal-drag numerical stability,
  multi-device validation, additional smooth-region gradient checks; 80% target
  and full objective are still incomplete.

## 2026-09-08 — standalone bulk heat fluxes and stable basal drag

- IN PROGRESS (@root): final correctness/coverage run, then separate commits.
- [x] Added heat-flux tests before changing source. Initial collection sample
  failed 3/3 imports on missing Veros; native JAX decorators/arrays preserve
  the existing immutable `state.settings` interface without fake dependencies.
  Initial scalar radiation/humidity/transfer checks then passed 25/25.
- [x] Extended CESM tests to hybrid pressure levels, isothermal hydrostatic
  height, iterative stable/unstable bulk exchange, masks and finite-difference
  temperature derivatives. Added dry/saturated/humid cases in both modules to
  verify nonzero latent-heat/water closure and exchange direction.
- [x] Latitude knot/midpoint checks failed 2/2: legacy scatter interpolation
  overwrote index zero for unmatched latitudes. Independent jnp.interp fixes
  both; net longwave docstring now matches its unchanged downward-positive law.
- [x] Basal drag stability/gradient tests: 27 failed, 39 passed before the fix.
  Replace log(exp(x)+1) with algebraically equivalent logaddexp(0,x); all
  66 cases then passed, including float32/float64, thresholds, disabled drag,
  and analytic thickness/velocity derivatives up to 90 m keel thickness.
- [x] Independent numerical review found no unintended changes. The review's
  saturated-only moisture gap was addressed before the final run.
- [x] Fast suite: 43 passed, 378 deselected. Tests pass Ruff and ty.
- Remaining limits: MITgcm ocean mask is currently unused by production fluxes;
  its diagnostic derivatives hold transfer coefficients fixed, so unrestricted
  AD is not their oracle. Legacy heat callers supply grav/radius; tests provide
  those constants explicitly because the current registry lacks them. Legacy
  setup/init still need removal of Veros dependencies and integration tests.

- [x] Full correctness suite: **421/421 passed in 43.40 s** on JAX 0.11.1.
  Basal stability 66/66, CESM heat flux 46/46, MITgcm bulk flux 27/27.
- Whole-package coverage **788/1515 statements (52.01%)**; both bulk heat-flux
  modules reach 100% statements. CI floor raised to 52%; no exclusions added.
- Test Ruff/format/ty checks pass. Source import sorting passes; the two legacy
  mixed-case heat-flux module names still trigger N999, retained for API stability.
- Target 80% remains incomplete. Next priority: standalone initialization and
  integration with artificial ocean masks, remaining boundary/gradient gaps,
  multi-device/GPU verification. Generated version metadata is still included
  in the reported whole-package denominator.

## 2026-09-08 — geometry and standalone coupled integration

- [x] Initialization tests first failed importing Veros. Native JAX arrays in
  the host initialization routine remove that unnecessary dependency.
- [x] Independent corner-area tests then failed 2/3: rAz used two neighbors
  divided by four. Restored the four-neighbor average; uniform grid area and
  nonuniform explicit-index geometry checks now pass.
- [x] Added tests before the new `veris.setup.artificial` example. An artificial
  central island blocks both staggered face directions. The host driver uses
  immutable state, full dynamic stress carryover, transport/cleanup and Growth,
  prescribed heat-forcing restoration, and periodic halo refresh.
- [x] Initial geometry/integration suite passed 8/8. Added forcing reset,
  nonzero stress, and fresh-process import checks following independent review.
- This is a serial demonstration with five EVP substeps, not a converged
  solution benchmark. Call initialize before distributed halo imports; the
  fresh-process test checks the supported standalone launch path.
- IN PROGRESS (@root): full correctness/coverage check and final lint checks.
- Legacy geographic Veros setup remains available but still unported/untested;
  the artificial example supplies the requested standalone coupled path without
  claiming equivalence to a full ocean simulation.

- [x] Full suite **431/431 passed in 52.43 s**; whole-package coverage
  **886/1567 statements (56.54%)**. Initialization 3/3 and artificial integration
  7/7 passed. New example, initializer, and tests pass Ruff/format/ty checks.
- CI floor raised to 56%. Advection and solver dispatch now reach 100% statement
  coverage through the coupled example. No coverage exclusions added.
- Remaining uncovered sources: generated version metadata (353 statements),
  legacy geographic setup (316), halo backend factory (10), model stub (2).
  More physical/gradient/distributed validation remains even for covered lines;
  the 80% goal and full objective remain active.

## 2026-09-08 — remove geographic setup and ocean-model dependency

- [x] User explicitly excludes the old geographic setup from testing and requests
  complete removal. Deleted its code, package initializer, asset manifest,
  documentation page/gallery image, unused plugin hook, and veros_fill setting.
- [x] Removed the Veros install requirement and setup discovery entry point.
  Package metadata now declares JAX >=0.11.1, NumPy >=2, Python >=3.12, matching
  the installed JAX minimum Python version. No Veros imports remain in veris.
- [x] Standalone documentation replaces plugin/copy-setup instructions; old
  documentation dependency removed. Historical author/copyright credit retained.
- [x] Built wheel and checked its file list, metadata and absent plugin entry
  points. Its requirements are only JAX and NumPy. Extracted-wheel integration
  passed in a temporary working directory outside the source tree.
- [x] Documentation build with latest compatible Sphinx/furo passes with warnings
  treated as errors. Repaired one title underline after the first strict build
  failed. Read the Docs now selects supported Python 3.14 and installs the package.
- [x] Full suite **431/431 passed in 52.03 s**. Coverage **886/1249 (70.94%)**
  after the user-requested deletions; this rise reflects removed obsolete code,
  not added test cases. CI floor raised to 70%, with no coverage exclusions.
- Remaining coverage gaps: generated version metadata (353 statements) and halo
  backend factory (10). Continue physical/gradient and distributed validation;
  the overall unit-test objective remains active.

## 2026-09-08 — final unit-test goal verification

- [x] User approved excluding generated `_version.py` from the 80% CI target,
  while retaining whole-package reporting. CI now saves full XML/JSON reports
  and separately runs `coverage report --omit=veris/_version.py --fail-under=80`.
- [x] Real four-CPU-device halo values and reverse-mode gradients passed for
  2x2, 1x4 and 4x1 meshes. The independent reference indexes global periodic
  coordinates and counts each input cell's copies for the adjoint. The initial
  gradient probe required entering JAX's explicit mesh context; no production
  change was needed. This does not certify multi-process/MPI or GPU execution.
- [x] Four nonlinear thermal sensitivity cases pass both central differences
  and the analytic implicit surface energy-balance derivative, away from caps.
- [x] Final full suite: **436/436 passed in 63.95 s**, no skipped/xfail cases.
- [x] Approved CI coverage gate passes: **886/896 maintained statements = 98.88%**.
  Whole-package coverage remains **886/1249 = 70.94%**, including generated code.
- [x] Tests pass Ruff, formatting, and ty. Whole-repository checks still report
  175 pre-existing/style findings and 39 typing diagnostics outside the clean
  test scope; logs saved in test_logs/ruff-final.log and test_logs/ty-final.log.
- [x] Independent final audit found no remaining test-harness/design/CI blocker.
  Earlier wheel integration and strict documentation checks passed after removal
  of all geographic setup code and Veros dependencies. Remote CI was not run.
- [x] Unit-test objective achieved: meaningful equation/conservation/reference,
  edge-case, gradient and coupled integration tests; deterministic fast mode;
  latest tested JAX environment; enforced >=80% maintained-code coverage.
- Future model work (not claims of this test deliverable): MITgcm bulk mask
  semantics, additional nonsmooth AD behavior, full multi-process reductions,
  GPU hardware verification, and broader repository style/type cleanup.

## 2026-09-08 — expanded validation (in progress)

- IN PROGRESS (@root): MITgcm masks, distributed reductions, and GPU validation.
  @nonsmooth owns additional gradient tests; @style_audit audits quality read-only.
- Verified two Tesla P100 16 GB GPUs with NVIDIA driver 580.173.02 outside
  sandbox. Sandboxed nvidia-smi cannot access driver. CUDA 12 is required for
  P100 architecture; JAX 0.11.1 CUDA dependencies are installing.
- Upstream MITgcm bulkf_forcing.F gates LANL calls with nonzero maskC;
  Veris currently ignores mask values. Plan records zero land diagnostics and
  safe inactive inputs, with unchanged wet-cell calculations.

### MITgcm masks and nonsmooth derivatives

- [x] Three land-mask regression cases first failed (finite, zero, and NaN land
  inputs). Safe inactive inputs and zero land diagnostics now pass, preserving
  wet-cell values and giving finite gradients with zero land sensitivity.
  The nonzero mask is a wet-cell selector, not an area weight. Upstream evidence:
  https://github.com/MITgcm/MITgcm/blob/master/pkg/bulk_force/bulkf_forcing.F
- [x] Ten nonsmooth cases check Superbee/ridging/clipping/floor selected AD
  linearizations, JVP/VJP adjoints, one-sided slopes, and the discontinuous
  thin-ice removal jump. Fixed an initial test collection error by selecting
  the serial halo backend before importing advection.

### Distributed reductions and hardware

- [x] Explicit mesh axes added to global_sum, EVP, and IceVelocities. Local
  totals exclude halos before psum; serial component dimensions are preserved.
  New API tests failed first, including dispatcher forwarding found in review.
- [x] Two real CPU processes pass sums/squared norms, JVP/VJP and poisoned-halo
  exclusion in 2x1/1x2 meshes and in 4x1/1x4/2x2 with two devices per process.
- [x] Two GPU processes (one P100 each) pass reduction and derivative oracles.
  Review fixed allocation handling to preserve inherited CUDA_VISIBLE_DEVICES;
  rerun with allocation 1,0 passes. GPU halo values/adjoints pass 2x1 and 1x2.
- [x] Full GPU suite: 455/455 cases passed with asserted GPU default backend.
  Later dispatcher addition: 4/4 direct/dispatcher sharded GPU cases passed.
  The initial CUDA-only run failed four print tests because JAX host callbacks
  require a CPU backend; corrected to JAX_PLATFORMS=cuda,cpu, with no numerical
  tolerance relaxation. requirements-gpu.txt records the P100-compatible extra.
- GPU worker shutdown emits JAX WatchTasksAsync connection-refused warnings
  after both workers report verified results; both processes exit zero. Logs
  retained in test_logs/gpu-halo-reductions-final.log. Multi-node networking/MPI
  launchers and a full distributed coupled integration are not certified here.

### Maintained code quality

- [x] Sorted/formatted maintained source, replaced dict constructors, removed
  unused bindings/imports, and made optional external mesh import explicit.
  Independent AST review found no hidden numerical changes in formatting.
- [x] Modernized doc/conf.py and setup.py, removed unused Click directive,
  corrected docutils error construction; strict Sphinx build and wheel build
  pass. Installed setuptools 84.0.0 for packaging checks.
- [x] Expanded CI Ruff/format/ty gate to maintained model, tests, documentation
  configuration and packaging. Those checks pass; public CESM/MITgcm names are
  retained. Generated version metadata and vendored sources remain untouched.
- Raw whole-repository reports improved from 175 to 131 Ruff findings and from
  39 to 24 typing diagnostics; remaining findings are generated/vendor code and
  the two preserved public module names, not claims of a clean raw repository.
- [x] Final CPU correctness/coverage and four focused commits completed;
  pushed to jax-only through 9539162.
- [x] Final CPU suite: **457/457 passed**, no skipped/xfail cases. Maintained
  coverage **895/904 = 99.00%**; whole package **895/1257 = 71.20%**. Approved
  80% gate passes, with XML/JSON whole-package artifacts retained. Maintained
  Ruff/format/ty checks pass. Existing remote baseline CI is green.
- [x] GitHub Actions run **34228786922** passed on **9539162**, including
  full correctness, maintained lint/type checks and the coverage gate.
  Expanded validation objective is complete within the documented local-machine
  scope; raw vendor/generated diagnostics remain explicitly reported above.

## 2026-09-08 — source typing

- [x] Annotated all project-owned code with explicit array,
  settings, state and return contracts; preserve numerical execution and PyTrees.
- User explicitly excludes generated _version.py and vendored Versioneer/Font
  Awesome internals. Use typed interfaces at their project-owned boundaries.
- @typing_audit reviews structural state/settings design read-only. Initial
  audit found nearly all model function parameters/returns unannotated;
  clean ty output currently reflects inference/dynamic boundaries, not complete
  public contracts. Mutable geometry initialization and immutable JAX state
  require distinct types; loop counts must remain integer settings.

- [x] Annotated every project-owned function signature: 435 definitions audited,
  including tests, probes, documentation hooks and packaging. Structural read-only
  domain protocols support caller-owned immutable state; explicit State/Settings
  named tuples preserve field order, replacement and JAX PyTree behavior.
- [x] Added typed JIT boundary retaining call signatures and lowering APIs. Direct
  JAX decorators erased signatures in the initial negative static test; a narrow
  cast of the same compiled object fixes this without a runtime call wrapper.
- [x] Review caught static settings lacking hashability, rejected boolean NumPy
  masks, and an unnecessary hashability constraint on uncompiled height constants.
  Positive/negative static checks now cover each; no blanket Any in model code.
- [x] Ship py.typed and generated-version interface; Versioneer is consumed through
  a project-owned stub. Wheel/sdist metadata and out-of-checkout valid/invalid
  consumer checks passed. Generated _version.py, Versioneer and Font Awesome
  internals are unchanged. CI enforces annotations only in maintained targets.
- [x] First full CPU run: 485/485 passed, maintained coverage 1038/1047 = 99.14%,
  whole-package coverage 1038/1400 = 74.14%. One later static regression verifies
  mutable height constants; final full CPU/GPU validation pending below.
- [x] Maintained Ruff, formatting and ty passed. Strict docs initially failed only
  because sandbox DNS blocked the Python intersphinx inventory; retry with network
  access requested. Numerical review found unchanged equations/indexing; the
  MITgcm output tuple is explicitly enumerated to retain its fixed return arity.

- [x] Final CPU correctness: **486/486 passed**; full GPU: **485/485 passed**
  before the final noncompiled-height static regression (then verified on CPU).
  Maintained coverage is **1037/1046 = 99.14%**. Strict Sphinx, final wheel
  and sdist, packaged consumer checks, Ruff/format/annotation gate and ty pass.
  Reviewer confirmed both final follow-ups; no remaining correctness blocker.
- [x] Committed/pushed refactor as 0c99dcd. First CI run 34233828681 exposed
  missing Sphinx in requirements-test.txt: the newly typed docs hook requires
  sphinx.application during static checking. Local Sphinx 9.1.0 had masked that
  dependency omission. Added that validated version to CI requirements; no
  source/runtime changes. Remote revalidation completed below.

- Dependency follow-up: Sphinx 9.1.0 requires Docutils <0.23. The local docs
  environment already uses 0.22.4, while the old test requirements pinned 0.23.
  CI 34234136636 failed resolution; aligned the pin to latest compatible 0.22.4.
  Package-index checks confirm Sphinx 9.1.0 is latest; local validated numerical
  and docs environments already contain this compatible pair.

- [x] GitHub Actions **34234265030** passed on **bce3b18**, including dependency
  installation, maintained lint/format/annotation coverage, ty, full correctness,
  coverage gate and artifact upload. Typing objective complete; generated/vendor
  internals remain unchanged. Local untracked AGENTS.md and test_logs preserved.

## 2026-09-08 — CPU/GPU profiling (in progress)

- [x] Oriented on jax-only at bce3b18. No previous profiling harness/results
  were present. Installed TensorBoard 2.21.0 and XProf 2.23.1 in .venv-latest;
  JAX/jaxlib remain 0.11.1 and pip check passes.
- [x] Verified two P100 16 GB GPUs, driver 580.173.02. Captured real CPU and
  single-GPU XPlane/Perfetto traces for a 64x64 artificial coupled step with
  400 EVP iterations. Twelve synchronized, unprofiled calls give baseline
  medians 186.404 ms CPU and 27.900 ms GPU; compilation is recorded separately
  as first-call elapsed time. Raw artifacts: test_logs/profiling/.
- [x] Exploratory whole-step JIT gives 183.739 ms CPU and 20.701 ms GPU;
  all 84 output fields match exactly at this input. No production code changed.
  GPU baseline has 98 executable dispatches per step. CPU host trace waits
  prevent attributing the large fill_overlap host duration to halo kernel cost.
  CPU speedup is unproven; GPU result requires larger/paired trials and AD checks.
- [x] TensorBoard localhost:6006 loads XProf and discovers all four traces;
  response saved in test_logs/profiling/xprof-runs.json. Live server tool session
  79737 (revalidate handle next turn). TensorFlow absent: some remote capture
  features disabled, local capture works. No browser rendering claim.
- Failed environment attempts: sandbox cannot resolve PyPI, access NVIDIA,
  or open TensorBoard sockets; approved external retries succeeded. Initial
  curl emitted gzip bytes; decoded response now saved as JSON.
- Next work and methodology: docs/superpowers/plans/2026-09-08-profiling.md.
  Need reproducible harness, larger/evolving workloads, actual CPU optimization,
  reviewed production changes, gradient/full CPU+GPU checks, and CI. Goal active.
- [x] Fresh CPU fast selection passes all 49 selected cases after installing
  profiler dependencies; git diff --check passes. No commit made this turn.
- IN PROGRESS (@benchmark_harness): test-first standalone paired benchmark CLI
  in benchmarks/ and tests/test_benchmark*.py. @root investigates CPU EVP and
  owns coupled-step production optimization/tests. Only one pytest runs at once.
- [x] Revalidated TensorBoard session 79737 live and converted all four XPlane
  captures through XProf: GPU kernel_stats and CPU hlo_stats JSON now saved
  beside the timing artifacts. This proves analysis conversion, beyond discovery.
- CPU affinity probe: pinning to one core increased whole-step time to 241 ms;
  reject as an optimization. CPU HLO partitions the large EVP velocity-update
  fusion across five tasks on this 48-logical-CPU host. Fusion-boundary and halo
  probes are exploratory; no numerical production change is accepted yet.
- [x] @benchmark_harness added paired fixed/evolving CLI, validation off-clock,
  per-variant XProf/Perfetto captures and metadata. Twelve focused tests passed;
  reviewed caveats documented in benchmarks/README.md. Baseline always means
  current Python step body, not historical kernels; artifacts need source patch.
- [x] Whole-step JIT tests first failed on missing compiled interface; typed JIT
  added to artificial.step with dynamic cooling. Three CPU tests pass: all-field
  evolving nonuniform/masked fixed+adaptive EVP equivalence, cooling JVP/VJP and
  central differences. Focused lint/format/ty pass. Full suite pending.
- [x] Paired 256x256 evolving GPU benchmark (12 calls, 400 EVP substeps) gives
  75.493 ms Python driver versus 69.372 ms compiled driver, paired ratio 1.090;
  exact equality of every field through all steps. Trace capture succeeded.
- CPU EVP probes: concatenate halos gave no convincing gain (190→188 ms);
  barrier on forcing worsened 190→203 ms. Barrier on drag plus stress divergence
  gave 190→118 ms at 64x64, but only 1.502→1.469 s at 256x256. Maximum absolute
  rounding difference 5.25e-11 after 400 steps; first strict exploratory comparison
  flagged a 1.15e-11 near-zero stress difference, not an accepted tolerance change.
- GPU EVP barrier probe (isolated after coupled benchmark ended): 17.924→18.492 ms
  at 64x64 and 66.058→54.451 ms at 256x256, exact output equality. Small-grid
  regression requires assessing combined benefit; do not claim universal gain.
- @evp_oracle captured pristine values/JVP/VJP for eight nonuniform rectangular
  fixed/adaptive/coastline cases with source hashes, before any EVP edit.
  @optimization_review performs independent read-only correctness/performance audit.
- [x] Inserted EVP drag/stress-divergence identity barrier after baseline oracles
  passed. All 11 focused CPU oracle/coupled tests still pass, including independent
  finite differences. No existing tolerance or iteration count was changed.
- [x] Independent review found no production correctness blocker. Fixed missing
  benchmark-test annotations; extended CI maintained checks to benchmarks/.
  Full maintained Ruff/format/annotation/ty checks pass. Reviewer suggested an
  np.savez allow_pickle change based on older NumPy; installed 2.5.3 signature
  explicitly supports it and fixture has exactly 16 expected keys, so retained it.
- Added requirements-profile.txt pinning validated optional visualization tools.
  IN PROGRESS: full CPU coverage suite, then GPU suite and final profiles.
- [x] Full CPU suite 509/509 passed outside sandbox; maintained coverage
  1041/1050 = 99.14%, whole package 1041/1403 = 74.20%. Initial sandbox run
  failed only because the existing two-process reduction test needs sockets.
- Artifact audit correction: XProf CPU hlo_stats conversion returns a valid
  but empty table for these host traces. CPU cost attribution uses Perfetto
  runtime fusion events and compiled HLO, not an empty XProf HLO statistics table.
  GPU kernel_stats contain nonempty kernel names/counts/timings.
- [x] Full GPU suite 509/509 passed with asserted GPU default backend
  (CPU-specific harness/reduction checks retain their intentional CPU target).
  No numerical tolerances were relaxed. No pytest remains running.
- IN PROGRESS: serial final original/optimized 64x64/256x256 CPU/GPU evolving
  matrix in tool session 30185. Original source archived from remote-matching
  76bf8bc; current source hashes and patch retained in test_logs/profiling/.
- Installed optional Perfetto Python API 0.58.2 for SQL analysis of final traces;
  existing JAX/numerical dependencies were unchanged.
- User explicitly requires this local node without scheduler; no jobs were
  submitted. A read-only scheduler query found this host unregistered; no more
  scheduler actions. Native host snapshot showed load 0.76 on 48 logical CPUs,
  both GPUs idle with no compute allocations. This is not historical load proof.
- Monitored native repeats (53 vmstat samples) showed median 95% idle,
  minimum 83% idle, no I/O wait/steal. Original64 CPU Python/JIT medians:
  134.9/163.1 ms and 137.5/164.8 ms. Current barrier CPU Python/JIT medians:
  158.3/151.9 ms and 165.5/148.9 ms. Thus default whole-step JIT would regress
  evolving CPU performance despite earlier fixed-state results. Acceptance held.
  Shared-node contention can affect timings but heavy host-wide contention was
  not present in these monitored repeats; frequency/cache effects remain possible.
- Do not conflate earlier sandbox/executed-source EVP probes with native final
  full-driver results. Shared cgroup/CPU limits match, but compiled fusion/launch
  contexts and workload differ. Hoisting basal keel/area factors alone gave no
  useful improvement in a further exploratory coupled probe; not implemented.
- Plan adjustment after performance evidence/review: preserve original Python
  step API and expose opt-in compiled_step for measured beneficial workloads.
  New interface tests updated first; production API change not yet made.
  @benchmark_harness investigates native same-process CPU barrier placements;
  root holds pytest while timing runs. Goal remains active; no commit yet.
- Native canonical (normally imported) uninterrupted CPU loops confirm the
  measurement method also matters, not just exploratory cloning. All48-core
  original driver: 211.78/171.15 ms across fresh reversed-order runs; one NUMA
  node's six physical cores: 133.76/134.65 ms; one socket's twelve physical cores:
  124.35/124.29 ms. All 84 final fields match exactly across all six runs.
  Topology/raw samples/NPZ outputs retained under native-affinity-*; these are
  local child-only affinity changes, never scheduler allocations or host changes.
- [x] Revised interface tests failed on missing compiled_step before adding the
  typed alias. Original Python step remains the default. CUDA-only EVP barrier
  uses jax.lax.platform_dependent selected by actual lowering target; CPU retains
  original fusion. Independent review found no AD/shard_map/type blocker.
- [x] Expanded EVP oracle to fixed/adaptive 400-step cases from hash-verified
  original snapshot, with unchanged FD/tolerances. All 27 focused CPU tests pass
  (12 EVP, 3 coupled, 12 then-current harness cases) after revised production code.
- [x] Added --validation final to avoid host validation between timed pairs;
  default each retains intermediate checks. New tests failed on missing API,
  then 18 harness tests passed including final-only mismatch/NaN and validation
  schedule coverage. Documentation states per-call synchronization remains.
- [x] Perfetto API 0.58.2 with SHA-verified v57.2 binary in .venv-latest/bin
  parsed saved traces and executed SQL. GPU256 original Python vs compiled
  barrier trace: 294 versus 3 executable dispatches over 3 steps. CPU trace
  analysis did not support the rejected blanket whole-step JIT default.
- IN PROGRESS: final revised full CPU suite (session72055), then GPU suite and
  final-source native measurements with matched validation schedules. No commits
  yet; previous 509-case validation predates the final API/per-backend revision.
- [x] 2026-09-08: Final revised full CPU and GPU suites both pass 519/519;
  maintained coverage 1041/1050 (99.14%), whole package 74.20%. Maintained
  Ruff/format/annotations/ty and pip dependency checks pass. No test remains live.
- [x] Final serial local matrix completed without scheduler use. GPU64 original
  Python/current compiled: 26.35/20.91 ms; GPU256: 74.67/57.48 ms. CUDA-only
  barrier has a small GPU64 cost but improves GPU256; compiled dispatch reduction
  gives net gains at both measured grids. All checked paired outputs are exact.
- CPU256 Python socket-local medians 1492.68/1460.65 ms versus all-affinity
  1618.25/1583.20 ms. CPU64 all-affinity varies 126.87 to 200.16 ms, while socket
  stays 168.42/171.36 ms. No universal small-grid gain or causal contention claim.
  Final 540 vmstat intervals: median idle95%, minimum76%, max I/O wait1%, steal0%.
  Standalone/paired CPU methods disagree; default_device context, resident JIT
  variants and interleaving remain possible causes, not established diagnoses.
  Follow-up for any CPU tuning: control those variables on the actual target loop.
- [x] Recorded accepted results, rejected approaches, provenance and limitations
  in benchmarks/RESULTS.md. Retained original CPU fusion/Python default; no
  speculative CPU optimization accepted. Final source hashes and patch retained
  alongside raw traces. Profiling code and reviewed CUDA/API changes are ready.
- [x] Independent final report review verified all twelve result rows and trace
  counts without blockers. Committed on jax-only: b286ce1 (profiling harness),
  b6aeb44 (opt-in compiled driver), 7aad88a (CUDA EVP/oracles). Source remained
  unchanged after final full-suite validation. Large local artifacts and the
  user-provided AGENTS.md remain untracked. No remote push performed.

### 2026-09-14 — Standalone reference cases

- Active goal: adapt dynamics/growth notebooks and parallel runner from the
  reference jax_halo_exchange branch into veris/setups; CPU jobs on aegir/v3,
  GPUs on the current node. Previous implementation absent in current checkout.
- IN PROGRESS: root owns dynamics/parallel and test scheduling; reference_audit
  reviews source fidelity. Plan: docs/superpowers/plans/2026-09-14-reference-cases.md.
- Source audit: dynamics uses fixed snapshot 15, 120 adaptive EVP substeps and
  600-second steps. Growth feeds returned Qnet/Qsw into subsequent days. Parallel
  reference imports missing generated initialize_dyn_1024; replace with CLI.
- [x] Implemented separate run_dyn/run_growth/run_parallel cases using existing
  initialization and kernels; reference-used forcing fields and sequence reviewed.
  Unused State fields intentionally retain shared VARIABLES defaults.
- [x] Initial serial tests 12/12 and parallel contracts 6/6 passed. Four-CPU
  single-process sharding matched serial after two steps on a rectangular grid.
- [x] Real aegir/v3 two-rank job 65253021 completed (1x2 mesh,12x16,2 steps,
  4 EVP iterations). Output agrees with serial CPU: maximum absolute error
  1.78e-15. Local GPU dynamics agrees within3.64e-12; full150-day growth
  CPU/GPU trajectories agree within2.28e-13. Artifacts: test_logs/reference-cases.
- Found and regression-tested nested output/nonfinite-output issues; fixed.
- AD validation: exact reference initial dynamics state yields NaN reverse AD
  already inside existing IceVelocities (before the new composition's transport).
  Wind forcing derivative is finite/matches FD; EVP velocity reverse derivative
  NaN versus FD0.101145633. Existing zero-norm singularities documented in
  test_evp_optimization; preserve reference physics, test new composition on its
  established smooth oracle. That smooth dynamics derivative test passes.
- Actual GPU parallel CLI exposed JAX explicit platform alias gpu expanding
  both CUDA and ROCm with strict initialization. Regression test reproduced;
  select cuda explicitly for the current NVIDIA node. No physics changes.
- [x] Final full CPU suite:731 passed,1 GPU-only test skipped; maintained
  coverage1709/1814=94.21%, whole-package1709/2167=78.86%. Dedicated GPU
  launcher regression then passed after CUDA selection fix. No pytest remains.
- [x] Corrected two-GPU parallel output matches serial CPU within4.17e-17.
  Runtime comparisons and source hashes saved in
  test_logs/reference-cases/runtime-validation.json. SLURM accounting confirms
  job65253021 COMPLETED exit0:0 on node453, partition aegir (constraint v3).
- [x] Maintained Ruff/format/annotation/ty checks and bash syntax pass. Sphinx
  rebuild with -E -W passes; initial sandbox-only inventory network warning
  resolved by native rebuild. Rendered page includes both scenario registries.
- [x] Independent final review found no blocker. Documented reference adaptations,
  CLI/local GPU/CPU queue usage, outputs, and pre-existing EVP AD limitation.
  Implementation complete on jax-only; local runtime artifacts remain untracked.

### 2026-09-14 — AD boundary repairs

- New user goal: fix zero-strain AD itself and related AD limitations.
- Root owns dynamics, regression tests and pytest scheduling; ad_thermo_audit
  audits thermodynamics and atmosphere. Plan: docs/superpowers/plans/2026-09-14-ad-boundaries.md.
- Initial scan found unguarded zero norms in viscosity, ocean/side/wind drag,
  adaptive EVP and free drift; inactive reciprocal branches in averaging/growth.
  Preserve forward equations and smooth derivatives, explicitly define finite
  origin linearizations for genuine norm kinks rather than hiding NaN results.
- [x] Reproduced all5 initial zero-state AD failures. Added exact-primal guarded
  norm sqrt with explicit zero origin linearization; zero-state tests plus
  float32/64 primitive tests and all12 EVP oracles pass (19 tests total).
- [x] Reproduced inactive slope-ratio and dry-corner reciprocal AD failures;
  guarded denominators before division. Corrected test oracle to35 wet cells,
  not48 storage-interior cells, because advection applies basin land masks.
- [x] Thermodynamic audit reproduced14 cases: absent ice/snow, zero salinity,
  calm winds in four atmosphere APIs, both precisions. Initial repairs plus
  existing thermo/transport tests passed204/205; sole failure was the wet-cell
  test-oracle error above. No tolerances relaxed.
- IN PROGRESS: free-drift Cartesian reformulation to recover actual nonzero
  Coriolis response at zero forcing (polar form loses it); nine new tests.
  Both zero forcing and zero mass-Coriolis remain a true square-root response
  singularity, to be tested/documented with an explicit finite AD convention.
- IN PROGRESS: masked CESM invalid land inputs, supported Area_reg=0 open-water
  behavior, capped stability square-root branches; reproductions added first.
  Full-State pullback and zero-strain shear-response regressions added after
  independent review requested stronger coverage than wind sensitivity alone.
- [x] Expanded focused run214/214 passed, including 25 thermodynamic cases, original thermo/free-drift oracles, explicit zero-strain
  shear response, and full-State pullbacks with fixed/adaptive EVP and both
  slip settings. Float32 weak-forcing test initially mixed float64 constants;
  corrected fixture dtype consistently rather than changing production casts.
- [x] Six further inactive-data regressions found4 failing cases (ice-free
  air temperature/humidity/wind and dry-column latitude); implemented targeted
  guards. Active-cell inputs and fractional-mask equations remain unchanged.
- [x] Sharded AD probe exposed run_parallel.remove_halos entering set_mesh
  inside a traced JVP/VJP. Explicit shard_map already owns its mesh; removing
  the nested context fixes it. Four-CPU and two-GPU evolving value/JVP/VJP/FD
  checks pass. Parallel contracts pass8/8 with1 GPU-only skip on CPU.
- [x] Independent review found no blocker. AD documentation now distinguishes
  finite selected origin/branch conventions from genuine classical derivatives;
  joint-zero quadratic free drift has a divergent one-sided sensitivity.
  Sphinx -E -W build passes. Full CPU suite currently running; GPU regressions
  and final checks/commits follow. No commits of these AD changes yet.
- [x] Final full CPU suite: 786 passed, 1 GPU-only test skipped. Maintained
  coverage 1756/1861 = 94.36%; whole package 1756/2214 = 79.31% (generated
  _version.py omitted only from maintained gate). GPU AD/EVP/precision suite:
  101/101 passed. All tests use final production sources; no pytest remains.
- [x] Additional nonuniform spatially weighted advection objective verifies
  JVP/VJP -1.26997263976 against FD -1.2699726315. This supplements total-mass
  testing, which alone cannot establish local derivative correctness.
- [x] Final maintained Ruff, format, annotation and ty checks pass. Production
  SHA256 hashes still match the full-suite snapshot. Evidence and documentation
  build retained in test_logs/ad-boundaries/. No forward tolerances relaxed.
- [x] All identified avoidable first-order AD failures repaired, including the
  actual stationary coastal State rather than substituting a smooth fixture.
  True norm/threshold conventions and the joint-zero free-drift degeneracy are
  explicitly documented and tested; forward physical thresholds are retained.

### 2026-09-14 — Math-formatted registry units in documentation

- [x] Replaced Type with Unit in Model settings and Physical constants tables;
  both tables read the corresponding registry entry's units field.
- [x] Wrapped all 87 dimensional units in Setting/PhysicalConstant registries
  with Sphinx math roles, grouping negative/fractional exponents and formatting
  degree symbols. All 179 entries retain defaults, types and descriptions;
  dimensionless `1` and `-` metadata remain unchanged.
- [x] Generated-document audit caught bare `-` being parsed as an empty list;
  table generation now renders dimensionless markers as literals.
- [x] Sphinx clean rebuild with warnings as errors passed. Inspected every
  generated Unit cell (34 settings, 145 constants), math nodes and MathJax HTML.
  Fast correctness suite passed (79 selected tests); Ruff, format and diff checks
  passed. Artifacts: test_logs/unit-columns/ and test_logs/unit-columns-*.log.
  Initial inventory-fetch warning resolved with an authorized network rebuild.
- [x] Follow-up: both Unit columns now display `-` for all dimensionless
  entries (including metadata `1`), per user request. Sphinx -E -W rebuild
  passed; all 179 generated rows verified, with 92 dimensionless cells showing
  a visible dash and dimensional math units preserved.
- [x] Pre-push full CPU correctness run: 785 passed, 1 GPU-only skip, and
  1 sandbox socket-permission failure. The affected two-process reduction/AD
  test passed unsandboxed (1/1), yielding 786 passing tests overall. Maintained
  coverage: 1756/1861 = 94.36%. No source changes were needed for the rerun.
  Full-run and rerun logs: test_logs/unit-columns-full-tests.log and
  test_logs/unit-columns-reduction.log. Ruff/format and diff checks passed.
