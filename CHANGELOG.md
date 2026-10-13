# Development log

## 2026-10-13 — Authoritative experiment framework

- [x] Tasks 1–7 complete; independent whole-branch review identified four
  Important findings, all reproduced and fixed in one pass: cold owned CPU/GPU
  rank checks, preset dtype before scenario normalization, recursive ownership
  of authored controls and shared island salinity bounds. Real borrowed
  services with two CPU devices per rank remain usable. No deferred findings.
- [x] Final reviewed-source CPU gate: 1,578 passed, four GPU-only skips,
  zero failures/errors in 963.098 s. Maintained coverage: 4,732/4,979 = 95.04%.
  The four skipped accelerator cases passed on native GPUs; reviewed-source
  GPU AD/CLI/two-device output gate passes 6/6. Fresh all-mode/custom hardware
  matrices pass 16 GPU cases and 16 one-/two-node aegir cases. Jobs `66304339`
  (one node, 54 s) and `66304340` (two nodes, 56 s) complete `0:0`; snapshots
  match serial at 6.04e-15 maximum scaled error and full histories match exactly.
- [x] Final Ruff/format/annotations/ty, shell syntax, strict Sphinx and clean
  wheel/sdist pass. Installed literal Python/CLI/config-export/mesh/JIT/AD
  examples pass. Scientific source and all three reference archives are
  unchanged. Evidence: `test_logs/framework/final-reviewed-*`,
  `review-final-contracts.xml`, `review-gpu.xml`, `final-review.json` and
  `hardware/aegir-review-equivalence.json`. Retain the intentionally interrupted
  pre-extension suite as partial evidence; it is not a passing gate.

- [x] Final native parallel CLI GPU check passes 1/1 with zero skips. Combined
  native GPU pytest gates pass 63/63, including every four CPU-skipped GPU case;
  scientific tolerances remain unchanged. Hardware matrices pass 16/16 cases.
- [x] Final Task 7 full CPU gate: 1,565 passed, four GPU-only skips, zero
  failures/errors in 959.369 s. Maintained coverage is 4,709/4,955 = 95.04%;
  the 80% gate passes. Ruff/format/annotations/ty and diff checks pass.
  Evidence: `test_logs/framework/full.{log,xml}` and `coverage.json`.
- [x] Additional native GPU initial-State/spatial-forcing JIT/grad/JVP/VJP/FD
  checks pass 2/2. Custom-ocean one-/two-GPU cases pass 4/4: snapshot maximum
  scaled errors 6.70e-16 (float64), 6.02e-7 (float32), with exact full histories.
  Both aegir placements match all 70 snapshot fields at maximum scaled error
  6.04e-15 and exact histories. Documented launcher `66303563` completes `0:0`
  in 6 s; its nine-field output matches serial at 3.68e-16 and 1200 s.
  Independent whole-branch review remains before final delivery.
- [x] Task 6 committed as `699d130`; authored/resolved exports, cold inspection,
  registered providers and executable custom-ocean blueprint are delivered.
- Task 7 acceptance: cold backend-policy regression failed before removing
  exclusive `jax_platforms` updates. Explicit CPU/GPU launches now preserve
  caller backend availability. Native one-/two-P100 matrix passes 12/12 cases,
  all 70 fields, both precisions and three physics modes: maximum scaled errors
  7.81e-15 (float64) and 6.97e-6 (float32). No numerical tolerance changed.
- [x] Local two-rank acceptance passes 11/11 in 142.212 s: all modes/custom
  geometry, both boundaries, spatial time forcing, instant/mean histories,
  borrowed-service reuse and owned cleanup after snapshot failure. Strengthened
  checks compare every history record and its bounds/sample metadata.
- Actual `aegir` matrix: one-node `66303413` completes `0:0` in 57 s; two-node
  `66303414` fails `1:0` on automatic hybrid mesh axes during spatial forcing.
  Reproducing axis regression failed, then explicit x/y axis construction fixes
  the root cause. Two-node rerun `66303463` completes `0:0` in 53 s. Both
  placements cover eight all-mode/custom/boundary cases; every snapshot/history
  record is independently compared with serial execution. Failed logs retained.
- [x] Native GPU scientific/startup/gradient/CLI/two-device-output gate passes
  60/60 with zero skips in 171.086 s. Strict Sphinx, Ruff/format/annotations/ty,
  shell syntax and scientific-source/reference hash audit pass. Three frozen
  archives are unchanged. Fresh wheel/sdist contain published blueprint/JSON;
  clean installed literal lifecycle/mesh/AD and Python/CLI/inspect examples pass.
  CI now runs the installed blueprint probe. Final full CPU gate/review remain.
- [x] @root: user approved the plan and native execution. Implemented
  the seven test-first units without further stage approval. Design and GPU/
  `aegir` test authorization remain in force; root owns the only pytest session.
- [x] Task 1 RED: 17 new cases reproduce stale scenario defaults, float32
  promotion rounding and absent grouped overlays. Implement immutable authored
  settings/constants/scenario/forcing plus separate resolved views; shared
  mapping merge now serves JSON and `Experiment.with_overrides`. Execution,
  initialization and restart/metadata consumers use resolved values.
- [x] Task 1 focused GREEN: 164/164 definition, overlay, preset, initialization
  and complete-restart cases pass, no failures/errors/skips (see
  `test_logs/framework/resolution-green.{log,xml}`). Source numerical equations
  and frozen references are unchanged. Final fast project sample is in progress;
  full verification precedes a later commit of the initialization ownership unit.
- [x] Task 1 final fast gate: 141/141 passed, no failures/errors/skips in
  242.391 s. Native task ledger records completion; no source unit is committed
  before the full configuration/initialization gate.
- Task 2 implements registered `SetupPreset` controls and regional field
  factories plus explicit copied/standalone/parallel launch profiles. Model
  resolution no longer branches on setup names. Profile overrides retain
  authored grid/zero-step requests and allow constants, initial fields and
  output defaults. Partial JSON grids inherit preset extents rather than
  prematurely resolving shared Grid defaults.
- Task 2 RED coverage includes 21 missing API cases, three omitted group/grid
  cases and three ownership/metadata defects. One freezing helper now owns
  nested control containers/arrays while preserving Parameter metadata. This
  prevents caller list/array mutation from changing future resolution.
- [x] Task 2 final focused 104/104 and fast 143/143 pass, no failures/errors/
  skips; maintained Ruff/ty pass. Evidence: `presets-final-green` and
  `presets-fast` logs/JUnit. Existing notebook captures retain their original
  checksums and all-field comparisons. Task 3 shared initialization is underway.
- Task 3 RED: eight provider/Growth/mask cases reproduce missing generic
  initialization and adapter bounds. Eleven additional ocean cases prove
  physical and precision checks must precede allocation. All setup-name branches
  are replaced by one regional provider pipeline; legacy setup initializers
  delegate to it. Current Growth physics and scientific archives are unchanged.
- [x] Task 3 focused GREEN: 126/126 provider, ocean, notebook, startup and
  adapter cases pass with no failures/errors/skips. The nonfinite Growth output
  test now injects at the shared initializer rather than the unused legacy
  adapter, preserving its rejection and no-output assertions. Expanded real
  sharding probe covers all four setup/geometry kinds on 1x4, 4x1 and 2x2,
  both boundaries (24 trajectories), including partition-crossing forcing.
  Full CPU/coverage gate follows; evidence is under `test_logs/framework/`.
- First Task 3 full gate: 1,439 passed, six failures, four expected GPU skips
  in 876.512 s. The expanded 24-layout sharding probe passes. Failed approach:
  routing array-only adapters through runnable clock/singleton-mesh restrictions
  rejects valid legacy initialization. An internal initialization-only definition
  now inherits shared validation while preserving those tuple-returning APIs;
  public Experiment/advance retain their execution checks. Unknown scenario
  parameters again raise TypeError. Focused regressions pass 71/71; an added
  sub-microsecond initialization regression and legacy invalid-step tests first
  fail, then receive narrow fixes. Full RED artifacts remain in
  `initialization-full-red.*`; a fresh full gate is required before commit.
- Task 4 RED: 41/41 new runner/runtime cases fail on missing interfaces.
  Implement validated DistributedExecution, explicit all-device profile requests,
  deep grouped execution overlays, ownership-aware resources, hybrid multi-slice
  mesh construction and public run_experiment. Pure physics and numerical
  State/Diagnostics remain unchanged.
- Additional RED/GREEN regressions cover cleanup after partial bootstrap failure
  and actual execution metadata in both snapshots and histories. RunResult adds
  separate immutable launch metadata. Explicit histories preserve authored
  schedules; absent schedules default to one instantaneous sample per step.
- [x] Task 4 focused gate: 145 passed, one expected GPU skip, zero failures/
  errors. Actual four-device all-field/spatial-forcing/output and two-process
  zero-/one-step runs pass. Unit ownership cases mock JAX policy together with
  bootstrap; real bootstrap policy runs in cold processes. Fix the probe's
  authored hourly versus per-step scheduling expectation, preserving a separate
  hourly-schedule regression. Evidence: `runtime-focused-final.*`.
- The narrowed legacy-only unknown-key error translation passes 111/111
  initialization contracts. Hold Task 3 completion/commit for the combined
  Task 3/4 full gate; all initialization, scientific, AD and new runtime cases
  must pass together before any code commit. No tests/oracles are removed.
- Combined Task 3/4 full gate: 1,452 passed, 41 failed, four GPU skips in
  809.403 s. All scientific/AD/sharding comparisons pass. Forty failures share
  the complete-restart reader's exact two-key execution schema check; expanded
  launch fields exposed it. One facade identity test requires the two approved
  public exports. Preserve `runtime-full-red.*` and fix these before commit.
- Added RED cases for reading the original version-one execution schema and
  cold rejection of unresolved launch controls. The restart reader accepts known
  optional launch fields, supplies only missing identity defaults for old files,
  and retains strict required/unknown-key and serial-layout checks. The facade
  identity test now explicitly verifies DistributedExecution/run_experiment
  against their canonical owners. Focused restart/resource verification follows.
- [x] Restart/resource regression gate: 117/117 pass, no skips/errors/failures
  in 259.838 s. All previous full-gate failures are covered. The real four-device
  runner probe additionally passes a supplied 2x2 mesh, all 70 field comparisons
  and a post-return calculation proving the caller's mesh remains usable.
  Maintained static/diff checks pass. A fresh combined full gate follows.
- [x] Combined Task 3/4 full GREEN: 1496 passed, four expected GPU skips, no failures/errors in 881.380 s. Maintained coverage clears 80%; frozen scientific, AD, 24-layout lifecycle, actual runner meshes/processes, restart compatibility/corruption and ambient-mesh contracts all pass. Native ledger records both units complete. Commit the verified ownership/runtime unit on `jax-only`; CLI unification and inspection/blueprint remain next. Evidence: `runtime-full.*` and `runtime-full-coverage.json`.
- [x] Ownership/runtime unit committed on `jax-only` as `10a5d88` after the
  1,496-pass/four-GPU-skip full gate and 95% maintained coverage.
- Task 5 IN PROGRESS: central/module CLIs delegate to public run_experiment;
  grouped --set parsing rejects duplicate/conflicting aliases and malformed JSON.
  Driver numeric defaults come from profiles, with explicit flag provenance;
  parallel zero-step runs now work. RED: 15/22 framework cases and 9/9 parser
  cases. Initial focused run exposes four calendar/sampling/error-message
  regressions; narrow fixes restore module per-step sampling without replacing
  authored streams and present --netcdf in the CLI error. Final focused pytest
  session is 50578 (`cli-focused-final.*`); verify it before a full gate/commit.
- Tasks 6/7 remain: authored/resolved exports and cold inspection, executable
  blueprint/installed documentation, final CPU/GPU/local-distributed/aegir
  validation, package checks and independent whole-branch review. Ignored native
  workspace contains prepared test drafts; do not count drafts as implementation.
- [x] Task 5 initial final focused: 87 passed, three expected GPU skips, no
  failures/errors in 76.399 s. Strengthened copied templates now execute real
  steps with all-field comparisons; additional custom-geometry and module-driver
  parity covers both boundary modes. The Slurm example delegates all-device
  mesh discovery/backend defaults to the parallel profile/shared runtime.
  Expanded equivalence tests are running (`cli-equivalence-final.*`).
- [x] Task 5 final focused: 81 passed, three expected GPU skips, no failures/
  errors. All copied templates execute real steps; Python/CLI parity includes
  custom geometry, both boundary modes and all 70 fields. Module-console invalid
  controls are concise and create no files. Required static/annotation/format
  checks pass; strict Sphinx was green before the console-only wrapper. Full
  adapter correctness/coverage verification now precedes its commit.
- [x] Actual two-node integration regression: aegir job `66302436` on
  `node453/node454` completes with ExitCode `0:0` in 48 s. All 70 physical fields
  match serial at maximum scaled error 2.50e-15; snapshot/history time is 600 s,
  and execution metadata reports two processes and 1x2 partitions. This directly
  verifies the hybrid mesh fix for the previous multi-slice failure. Evidence:
  `hardware/aegir-two-equivalence.json` and job log under `test_logs/framework/`.
  Final packaged/GPU/blueprint acceptance still remains.
- [x] Task 5 full GREEN: 1536 passed, four expected GPU skips, no failures/errors in 926.569 s. Maintained coverage clears 80%; strict docs/static/annotation/shell checks pass. Native task ledger marks the adapter unit complete. Commit on `jax-only`, then proceed to authored/resolved export and cold inspection/blueprint. Evidence: `cli-full.*` and `cli-full-coverage.json`.
- [x] Task 5 committed as `bcd5d29` after 1,536 passing tests/four GPU skips
  and 95% maintained coverage. Task 6 begins with authored/resolved export and
  cold inspect/export regressions; no production implementation yet. The
  custom-ocean blueprint remains a prepared draft until installed execution
  proves paired Python/CLI equivalence. Final GPU/packaging/review gates remain.
- Task 6 RED covers missing sparse/pinned modes, cold inspect/export, profile
  identity/version selection, partial output provenance, partial profile bootstrap
  and missing blueprint source. One codec now exports authored/resolved views;
  version 2 represents active profile/distributed/all-device declarations while
  version 1 remains supported. Shared execution decoding deeply merges profile
  controls; old checkpoints normalize only missing identity defaults.
- [x] Task 6 focused GREEN: 126/126 pass, no failures/errors/skips, 253.490 s.
  Strict Sphinx/static checks pass. The custom-ocean factory cold test passes.
  A clean installed wheel executes the exact published Python/CLI/inspect blocks:
  all 70 physical fields and authored/resolved configs match at 1800 s. Evidence:
  `installed-framework.log`, `exports-profile-green.*` and package/docs artifacts.
  Published authored/resolved growth JSON examples are included in the sdist;
  final focused/full gates remain before committing this delivery unit.
- Final GPU matrix exposed a runtime policy issue before scientific comparison:
  the first CPU launch narrows `jax_platforms` to CPU, so a subsequent GPU request
  in that interpreter cannot use available P100s. Preserve the failure log and
  partial CPU reference. A cold policy regression is prepared; fix after the
  active Task 6 full gate finishes, then rerun GPU acceptance with fresh paths.
  Do not treat GPU matrix startup as successful GPU evidence.
- [x] Task 6 full GREEN: 1554 passed, four expected GPU skips, no failures/errors in 916.686 s. Maintained coverage clears 80%; strict docs/static/annotation and clean-installed literal blueprint proof pass. Commit the export/inspection/blueprint unit on `jax-only`. Known CPU-then-GPU platform narrowing is explicitly tracked for Task 7; GPU acceptance and final goal completion remain unproven.
- Scope: fulfill the pasted consolidation request on `jax-only`;
  audit base is `9118648ade93de72c696c68c6067fabb5b21391d`. Read AGENTS.md,
  DESIGN.md, CHANGELOG.md and the pasted requirements. GitHub confirms the
  requested branch. Existing untracked `test_logs/` remain local evidence.
- [x] Audited definition/codec, low-level allocation/ocean adapter, all three
  setup initializers, central/copied/module CLI entry points, parallel launch,
  Simulation and pure timestep/rollout. Specialized read-only default-ownership
  and test-quality reviews are complete; neither started pytest or edited files.
- [x] Host probe reproduces a precedence defect: replacing island scenario
  timestep/EVP/cooling with 300/7/25 leaves resolved controls at 600/5/100.
  Preset resolution currently writes inherited defaults into explicit inputs.
  A second host probe finds float32-to-float64 promotion retains rounded
  inherited island salinity/growth thickness. Do not patch individual CLI flags;
  preserve authored values separately and resolve the full overlay chain.
- [x] Identified setup-name dispatch duplicated between serial/sharded
  initialization and runtime resources owned by the parallel example. Current
  builtin, copied-example and driver grids/durations differ; preserve these
  deliberately through explicit profile/experiment values with one owner.
- [x] Prepared and self-reviewed the architectural proposal in
  `docs/superpowers/specs/2026-10-13-experiment-framework-design.md`, including
  complete entry-point inventory, defaults, raw/resolved configuration semantics,
  provider/runner contracts and requirement-by-requirement verification gates.
  No production or test sources changed. Explicitly invoked Superpowers requires
  written-spec and implementation-plan review before implementation; proposal
  commits still obey the repository's full-suite gate.
- User authorizes native local GPU verification and parallel execution locally
  or through Slurm on `aegir`. Required evidence will distinguish local meshes,
  separate processes and actual multi-node execution.
- [x] Fresh deterministic fast CPU baseline: 139 passed, no failures/errors or
  skips, 433.029 s in JUnit. The only pytest session is terminal. Artifacts:
  `test_logs/framework-audit/baseline-fast.{log,xml}`. Previous full results
  remain baseline history, not evidence of completing this consolidation.
- [x] User approved the written design; prepare its implementation plan next.
  Unsandboxed device inspection confirms two available Tesla P100 GPUs. Slurm
  reports `aegir` up with 20 idle nodes; the partition supplies CPU nodes.
- [x] Wrote and self-reviewed
  `docs/superpowers/plans/2026-10-13-experiment-framework.md`: seven test-first
  units cover raw/resolved precedence, registered providers/default owners,
  shared initialization, execution resources/runner, all CLI adapters,
  authored/resolved export/blueprint, and complete scientific/GPU/distributed
  verification. Plan review and execution-method selection remain pending;
  production/test sources are unchanged. Recommend native implementation with
  a final independent review because interfaces and pytest ownership are coupled.
- [x] Authorized launch baseline: Slurm job `66298493` on `aegir`, one node
  (`node453`), two CPU ranks, 8x12 dynamics grid, one 600 s timestep/two EVP
  iterations completes with scheduler ExitCode `0:0` in 35 s. All nine selected
  physical outputs match the serial run (maximum scaled error 8.12e-17); both
  snapshot and final instantaneous history have time 600 s. Evidence:
  `baseline-aegir-one-equivalence.json` under `test_logs/framework-audit/`.
- [x] Found a real multi-node correctness blocker before implementation:
  otherwise equivalent two-node job `66298494` (`node453`, `node454`) fails
  with ExitCode `1:0` in 33 s. Both ranks reach `run_parallel.create_mesh`, then
  installed JAX 0.11.2 `sharding_impls.make_mesh` rejects devices spanning more
  than one `slice_index`. Its hybrid-mesh API supplies the relevant construction
  contract. Task 4 now explicitly includes this failing topology/regression;
  no production patch has been made while plan review is pending.
- Native GPU baseline initially passes both central CLI cases but the parallel
  CLI subprocess aborts after CUDA OOM (3.74 GiB allocation) while its GPU parent
  retains default memory reservations. Devices return to 3 MiB used after exit.
  [x] Identical three cases pass with `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  as recommended by JAX's GPU allocation documentation; the failure log remains
  intact. This is a verification-environment change, not a numerical code fix.
  Logs/JUnit: `baseline-gpu*` under `test_logs/framework-audit/`; combined result
  is `hardware-baseline-summary.json`. All launched jobs/processes are terminal.

## 2026-10-13 — Centralized Click command line interface

- [x] COMPLETE (@root): add the installed `veris` command and corresponding
  documentation on `jax-only`. Current local/remote HEAD is `dfb5313`; the
  checkout has no tracked modifications. Existing `test_logs/` are local
  evidence and must not be staged wholesale.
- Read DESIGN.md and AGENTS.md. Inspected Vercor's current CLI documentation:
  `show-setups`, `copy-setup`, `run`, version/help, logging and precision.
  Its documentation is accessible with curl; the web page opener fails.
  GitHub's `nutrik/vercor` `jax-only` tree returns 404; no obsolete branch
  is used as a source reference.
- Proposed a bounded CLI extension using existing Experiment/Simulation APIs,
  copied Python preset templates, native JSON definitions, CPU/GPU/mesh and
  output controls, and `VERIS_SETUP_DIR` discovery. The user explicitly approved
  this design before implementation. Work stays in the requested `jax-only`
  checkout; Click is already a package dependency.
- Read-only test-quality/documentation role (@cli_contract_audit) reviews
  integration contracts and packaging requirements. Root owns the only pytest
  session. Use the established `.venv-latest`, not the absent `.venv`.
- [x] Tests precede the CLI implementation: initial RED is the missing
  `veris.cli` module. First focused run: 34 passed, one CPU-expected GPU skip,
  one assertion-wording failure (`divide` versus existing `divisible` error).
  Check the error's partition context instead. Separate reference trace-level
  RED rejects `-v trace`; add it as logging level 5.
- [x] Implemented group/module entry point, deterministic template registry,
  non-overwriting directory copies, factory/JSON loading and existing lifecycle
  execution, shipped templates and console-script packaging. Explicit precision
  overrides update settings and serialized physical-constant dtype together.
- [x] Added CLI documentation with generated current help and links from Usage,
  quickstart, setup gallery, README and package hierarchy; DESIGN.md records
  ownership. Read-only implementation reviews are underway (@cli_code_review,
  @cli_contract_audit).
- [x] Independent reviews: no remaining required findings. Review reproducer
  confirms omitted distributed backend reached ambiguous initialization (RED).
  Require explicit CPU/GPU backend for distributed launches before initializing
  resources; preserve omitted backend for local definitions. Expanded focused
  CLI suite: 39 passed, two CPU-expected GPU skips (31.91 s).
- [x] Clean wheel installation smoke executes version, setup discovery, copy
  and one-step float32 growth outside source paths. Package-focused run:
  37 passed, two CPU-expected GPU skips (46.65 s); strengthened later CLI checks
  additionally execute a copied coupled island and omitted-option JSON output.
- [x] Maintained lint, formatting (201 files), annotation checks and ty pass.
  Strict Sphinx passes after correcting two heading underline lengths; generated
  help asserts command success. Initial docs warnings were formatting only.
- [x] Full CPU correctness: 1382 passed/four expected GPU skips, zero failures,
  789.82 pytest seconds/793.60 external seconds. All four skipped cases are
  subsequently verified on native GPUs. Maintained coverage is 4199/4449
  (94.38%); the 80% gate passes. Whole-package coverage includes generated
  version metadata (4340/4802, 90.38%). Evidence: `test_logs/central-cli/`.
- [x] Native GPU checks: four passed, no failures/errors/skips, 33.40 pytest
  seconds/36.70 external seconds on two Tesla P100s with JAX 0.11.2/CUDA 12.
  Cover central serial growth and two-GPU dynamics with float32 snapshots,
  existing two-GPU scheduled output and parallel-driver finite output.
  Initial sandbox attempt cannot initialize CUDA (`CUDA_ERROR_NO_DEVICE`);
  explicitly user-authorized GPU escalation succeeds. Do not diagnose this
  sandbox visibility failure as a model/runtime regression.
- [x] Final strict docs and maintained tooling checks pass. Fresh wheel/sdist
  build succeeds; clean installed wheel executes all three copied presets for
  one real CPU timestep outside the source checkout, with finite float32 fields
  and correct absolute times. Editable reinstall registers the local command.
  Existing setuptools deprecation notices are unrelated to this feature.
- Deliver this cohesive CLI feature on the requested `jax-only` branch after
  all required checks pass. Scientific kernels, reference data and tolerances
  are unchanged; local logs/build artifacts remain unstaged.

## 2026-10-12 — Investigate GitHub correctness failure #30

- [x] COMPLETE (@root): use GitHub CLI to diagnose and fix hosted correctness
  failures after completing the calendar goal. Read-only harness/performance role
  (@workflow_audit); root owns pytest and GitHub run coordination.
- Run #30 (`38257785337`, commit `44c41ba`): dependencies, lint, formatting,
  annotations and types pass. Full correctness reports one `F` at 82% before
  being cancelled at 20 minutes at 98%; cancellation prevents the traceback and
  no artifacts were uploaded. Removing the cap alone is not proof of a fix.
- Pushed verified commits `1537329` and `a8b5601` to `jax-only`. Uncapped run
  #31 (`38262562125`) completed: 1335 passed, two expected GPU skips, three
  failures in 1145.03 seconds. The failing closed-y, lifecycle and direct-advance
  sharding probes all abort in XLA's in-process CollectivePermute rendezvous
  (SIGABRT, expected four participants/three arrived, or two/one). All are on
  worker gw0. These are native collective failures, not numerical discrepancies
  or the removed 20-minute workflow cap.
- GitHub CLI log cache initially fails in read-only `/groups/.../.cache/gh`;
  set `XDG_CACHE_HOME=/tmp/veris-gh-cache` for log inspection. No credential
  changes. Failed logs are retained in `test_logs/github-ci/`.
- Pinned-CI standalone reproduction: four logical CPU devices restricted to one
  permitted CPU pass all rollout fields and AD assertions in 419.86 seconds.
  The existing pytest wrapper has a 360-second child deadline; reproduce its
  timeout through the exact test before adjusting that deadline. The exact
  wrapper reproduces TimeoutExpired at 360 seconds. This is a separate resource
  sensitivity: that test passes in hosted #31, so do not change its deadline
  without evidence that the actual hosted fix still requires it.
- User requests JAX >=latest 0.11.2 for the entire project and CI. Raise the
  package minimum to 0.11.2 and align CPU JAX/jaxlib and CUDA 12 pins to 0.11.2;
  retain historical environment/reference provenance. Compare actual failing
  sharding probes under narrow CPU affinity before claiming a runtime fix.
- Fresh CI-pinned 0.11.2 environment resolves and passes dependency, lint,
  format, annotation and type checks. Strict Sphinx, wheel/sdist and clean
  installed lifecycle examples pass; wheel metadata requires JAX >=0.11.2 and
  contains all 65 modules. CI now reports JAX/jaxlib versions and permitted CPUs.
- The two-device direct-advance probe passes all fields/JIT/AD on one permitted
  CPU with both JAX 0.11.1 (161.10 s) and 0.11.2 (174.89 s). Restricted affinity
  alone does not reproduce the hosted native abort on this machine. Full CPU
  correctness/coverage and restricted four-device probes are in progress.
- Independent read-only dependency/workflow review (@final_review): no findings.
  The CPU collective setup is unchanged between these JAX Python releases;
  cross-process gloo configuration is irrelevant to the failing in-process
  communicator. Do not claim the upgrade fixes CI before a hosted green run.
- Existing native GPU evidence remains applicable: all 30 GPU checks passed
  earlier today on the unchanged scientific source/tests with JAX, jaxlib,
  CUDA 12 plugin and PJRT all at 0.11.2. Confirmed these four installed versions
  match the updated GPU requirement and dependency checks remain clean.
- [x] Fresh full CPU correctness with the updated CI pins: all 1340 unique
  cases, 1338 passed/two expected GPU skips, 779.50 external seconds; maintained
  coverage remains 4056/4289 (94.57%). All three hosted failing probes pass when
  workers have sufficient CPUs. Strict tooling checks remain green.
- Actual resource RED on JAX 0.11.2: the lifecycle probe with four logical
  devices restricted to two permitted CPUs aborts after 225.56 seconds, matching
  hosted XLA CollectivePermute (four participants/only three arrived). The
  version upgrade alone is insufficient. The closed-y probe passes both coupled
  slip cases under the same two-CPU limit in 370.39 seconds. Test lifecycle with
  four permitted CPUs to confirm a sufficient allocation before adjusting CI.
- [x] Resource GREEN: the identical lifecycle probe on JAX 0.11.2 with four
  permitted CPUs passes all 20 preset/custom initialization, run and output
  cases in 217.71 seconds. This isolates the failing worker CPU allocation.
  CI now uses one worker, retaining loadscope, affinity partitioning, no worker
  restart, every test/coverage assertion and the uncapped correctness step.
  Local three-worker HPC runs remain available. The full correctness suite above
  verifies the unchanged source/harness under the updated dependencies.
- [x] Committed and pushed the dependency/allocation fix as `e5bb87b` on
  `jax-only`. GitHub run #32 (`38268800211`) passes every step: full CPU
  correctness, coverage, lint/format/annotations/types, strict documentation,
  wheel/sdist, installed public lifecycle examples and artifact upload.
  Hosted logs confirm JAX/jaxlib 0.11.2 and four permitted CPUs `[0, 1, 2, 3]`.
  Result: 1338 passed/two expected GPU skips in 1332.97 seconds (22:12);
  downloaded coverage confirms 4056/4289 maintained lines (94.57%). The test
  step continues beyond the former 20-minute cap without cancellation.
- Hosted verification: https://github.com/team-ocean/veris/actions/runs/38268800211.
  Full logs, status, coverage artifact and independently checked aggregate audit
  are retained in `test_logs/github-ci/`. Scientific source/harness/reference
  hashes remain unchanged; no numerical tolerance or child deadline was relaxed.

## 2026-10-12 — Remove explicit CI correctness timeout

- [x] COMPLETE (@root): remove `timeout-minutes: 20` from the full CPU
  correctness step and rename it `Full CPU correctness`; keep the complete
  test selection, three-worker maximum and coverage gate.
- [x] Reconcile the active runtime descriptions in DESIGN.md, tests/README.md
  and tests/AUDIT.md. Retain dated historical targets and timing evidence.
- Workflow YAML parses and the full correctness/coverage command is preserved.
  Verification shares the exact source/test tree of calendar commit `1537329`:
  1338 CPU passes, two subsequently verified GPU skips, 30 native GPU passes,
  94.57% maintained coverage, clean lint/format/annotations/types, strict docs
  and clean package/installed examples. No 20-minute assertion is added.
- Local evidence is under `test_logs/calendar-averaging/`. The user reports
  the timeout is already resolved remotely; the following task investigates
  the separate failed GitHub run using GitHub CLI.

## 2026-10-12 — Calendar averaging verification

- [x] COMPLETE (@root): preserve correct averaging and add independent checks
  for all months of Gregorian, 360-day and no-leap years on `jax-only`.
- Baseline `44c41ba`: current calendar arithmetic already selects twelve 30-day
  months for `360_day`, `_MONTH_LENGTHS` for `noleap`, and actual Gregorian
  dates. The user confirmed the reported defect was a misunderstanding and
  supplied no failing output. Preserve the correct production behavior.
- Independent roles: calendar/test quality and code audit (@calendar_audit),
  workflow/documentation/performance configuration audit (@workflow_audit).
  Root owns the only pytest session. Calendar audit passed 675 standalone
  boundary checks; existing monthly compiled tests chiefly cover February and
  share the host calendar oracle. Add independent all-month output expectations.
- Use the existing `.venv-latest` (Python 3.14.7); `.venv` is absent. Remain in
  the requested `jax-only` checkout. Existing untracked `test_logs/` are local
  evidence and must not be staged wholesale.
- [x] Added 28 independent cases with literal month lengths through two years;
  real host and compiled quadratic means, sparse samples, simultaneous annual
  streams, century/leap rules, year transitions, partial windows and split-run
  continuation. Counts, bounds, midpoints and calendar metadata are exact.
- Initial all-month run: 26/28 cases pass. Two compiled 360-day cases differ
  from the scalar oracle by one float64 ULP because JAX division can multiply by
  a reciprocal. Exact array equality is an unsuitable numerical requirement;
  allow two ULPs with aggregate error reporting, retaining exact counts/bounds.
- [x] All 142 focused calendar/host mean/planner/compiled mean checks pass,
  including all 28 new independent cases. Types, annotation and workflow YAML
  checks pass. Initial lint requests `itertools.pairwise` for adjacent edges;
  adopt it before final verification.
- [x] Three wrong-calendar mutations are rejected: no-leap using 30-day months,
  360-day using ordinary month lengths, and Gregorian using a fixed month width.
  Mutations are in-memory probes only; production files remain unchanged.
- [x] Full cold CPU correctness: 1338 passed, two expected GPU skips, all 1340
  unique cases; external wall 790.46 seconds. Maintained coverage: 4056/4289
  lines (94.57%), passing the 80% gate. All 28 new cases pass in the full run.
- [x] Native GPU verification: all 28 new cases plus both GPU-only contracts
  pass on two Tesla P100 devices, with no skips or failures (30/30, 52.38 s).
- [x] Lint, format, annotations, types and dependency checks pass. Strict Sphinx,
  wheel/sdist and clean installed public lifecycle examples pass. Archives retain
  all 65 modules and the new test file; all 199 recorded source/harness/reference
  hashes stay unchanged during verification. An initial package audit wrongly
  expected GitHub workflows in the sdist; retain the manifest's existing scope.
- [x] Independent whole-change review (@final_review): no Critical, Important
  or Minor findings; literal expectations and two-ULP mean tolerance verified.
- Evidence: `test_logs/calendar-averaging/` contains full and GPU status/JUnit,
  coverage, mutation results, source hashes, docs/package and installed logs.
- Follow-up requested by the user: after this work is committed, investigate
  GitHub's failed "Remove legacy compatibility modules and wrappers #30" run
  using GitHub CLI and fix CI. The user confirms the remote timeout is resolved.

## 2026-10-12 — Remove compatibility scaffolding

- [x] COMPLETE (@root): removed legacy source paths, setup timestep/output
  wrappers, corresponding alias tests and obsolete migration documentation.
  The user's explicit cleanup request supersedes the previous compatibility rule.
- Baseline: `f2e14b6` on `jax-only`; 38 module shims plus `veris.settings`,
  model.types schema reexports, setup timestep wrappers and parallel output
  helpers identified. Last full suite: 1393 passes, two GPU skips, 783.11 s,
  maintained coverage 94.64%. Root owned the single coordinated pytest session.
- [x] Architecture RED: seven canonical-only ownership/export cases fail on
  the compatibility layer. GREEN: all 28 architecture/schema/import-order cases
  pass after removing 39 flat modules and obsolete setup/CLI helpers.
- Failed caller migration: scheduled output received the new (State, Diagnostics)
  carry with a State-only field selector. Both precision cases reproduce the
  mismatch; select State from the canonical carry before field selection.
- Source AST audit: 191 retained definitions unchanged apart from imports and
  docstrings; changes are advance dispatch folding, CLI option removal and the
  island factory's error text. All reference files retain baseline hashes.
- Failed tooling attempt: system Python3.9 cannot parse Python3.14 type aliases.
  Source edit script stopped after shim deletion; rerun only with activated
  .venv-latest and idempotent deletions. No numerical source body was altered.
- Ruling: scientific settings such as explicitDrag predate module restructuring
  and remain part of the frozen configuration schema; their removal would be an
  unrelated model-schema change. Keep scientific metadata and parameterizations.
- [x] Final collection: 1312 cases. Deleted 86 module alias/old
  pickle/reexport cases and added seven canonical ownership cases (net -79),
  plus removed one schema reexport case, one obsolete shared-type-file case and
  one duplicate unequal-timestep wrapper case and one obsolete save_output-only
  contract (total net -83 from 1395). Scientific compiled/CLI cases
  move to execution/test_compiled_advance and setups/test_lifecycle_cli;
  physics/test_unequal_timesteps checks all 75 arrays in eager and JIT paths.
  Seven frozen trajectories retain exact settings and pure/host all-field checks.
- [x] Full lint, format (189 Python files/Markdown code blocks), annotation and
  types pass. Strict Sphinx and dependencies pass. Independent whole-change
  review has no remaining Critical/Important findings; corrected an unnecessary
  archived settings filter before full verification. No scientific tolerance,
  dtype or derivative direction was removed.
- First full run found 13 stale harness contracts: twelve dynamically assembled
  old module names in annotation tests and an obsolete factory error-message
  match. Updated all twelve to canonical owners and the forcing message, retaining
  every numerical and annotation assertion before a fresh complete run.
- User clarification: no test should assert a 20-minute full-suite duration.
  None exists. As explicitly requested, double only the four-CPU closed-wall
  scientific probe's subprocess timeout from 1200 to 2400 seconds; retained its
  value and gradient assertions. External runtime measurements and CI stay separate.
- One further first-run failure was a migrated obsolete save_output-only test
  that incorrectly demands finite-value validation from low-level write_snapshot.
  Removed that obsolete helper contract; the maintained parallel CLI test retains
  rejection before file creation via canonical Simulation.snapshot. No new storage
  behavior was introduced.
- First full result: 1297 passed, 14 stale contracts failed, two expected GPU skips,
  784.09 s external wall (781.52 s pytest). All 14 failure reasons are confirmed in
  first/full.log; corrections above affect only harness paths/messages and remove
  the obsolete helper-only test.
- [x] Final cold CPU correctness: 1310 passed, two expected GPU skips, all 1312
  cases exactly once. External wall 756.48 s (12 minutes 36 seconds; pytest
  754.64 s), with three module-grouped workers, disjoint CPU placement and fresh
  private compilation caches. Both skipped contracts pass separately on the two
  Tesla P100 GPUs (14.66 s external wall); no errors or skips in the GPU run.
- [x] Maintained coverage: 4056/4289 lines (94.57%), preserving all 4026 unchanged
  previously covered lines with zero unchanged coverage losses. All 181 final
  source/test/benchmark Python hashes and ten numerical reference-file hashes
  remain unchanged through verification. Generated version metadata retains its
  existing exclusion from the 80% coverage gate.
- [x] Clean wheel/sdist contain all 65 Python modules and none of the 39 obsolete
  files; 245 source-archive files are verified. Clean installed literal README
  and documentation examples pass, including CPU sharding, restart and JIT/AD.
  Strict Sphinx, lint, format, annotation, type and dependency checks all pass.
- [x] Final hierarchy: model, grid, physics (dynamics/transport/thermodynamics),
  numerics, execution, io and setups; canonical owners are documented in
  docs/reference/package-hierarchy.rst. Maintained modules remain at most 786
  lines. Exact scalar/JUnit/coverage/source/package evidence is retained under
  test_logs/cleanup/. Deliver the verified cleanup on jax-only.
- Roles: source/deduplication audit (@structure), test quality and canonical
  caller migration (@performance), documentation cleanup (@documentation).
- Ruling: retain the current root/io convenience API and preset initialization
  factories because canonical lifecycle and installed examples use them; remove
  old-path/module/step aliases. Preserve immutable numerical reference bytes;
  the unequal-timestep oracle tests explicit scientific kernel composition,
  with no bypass of the public coupled-clock validator.
- Ruling: execute in the user-designated `jax-only` checkout and complete all
  checks before one cleanup commit. User authorization and autonomous execution
  take precedence over skill approval menus and intermediate commits.

## 2026-10-12 — Scientific package hierarchy and 20-minute test budget

- [x] COMPLETE (@root): implement the current AGENTS.md hierarchy goal on
  `jax-only`; root owns all pytest runs, canonical moves and compatibility.
- Read DESIGN.md, latest validation/runtime history and current source/API/CI
  inventory. Current maintained baseline: 4058/4296 lines (94.46%), 1273 CPU
  passes plus two subsequently verified GPU cases; last full wall 2219.44 s.
  Existing runtime improvement does not meet the new 1200 s requirement.
- Independent read-only roles: performance/test harness (@performance),
  dependency/deduplication/code quality (@structure), documentation/packaging
  (@documentation). No concurrent pytest or production edits by auditors.
- [x] Migration contracts written first: all 94 migration cases fail on missing
  canonical owners; geometry/arrays/shared halo cases also fail before moves.
  GREEN after 38 source moves plus geometry/array ownership and shared packed halo
  extraction: 97/97 pass in 20.17 s, including legacy pickle globals, module/patch
  identity, root exports, fresh import orders and real halo JIT/AD.
- [x] Regrouped 89 existing tests by responsibility; all original collection
  counts retained (1372 cases including 97 new migration contracts). Initial
  fast development selection passes 138 cases in 550.74 s. Corrected two moved
  harness paths/imports identified by types and independent review.
- [x] Strict Sphinx passes with new hierarchy/migration page and canonical API
  references. Installed public example blocks remain unchanged.
- [x] Shared halo tracing and inherited subsecond private cache regressions fail
  before implementation; actual dtype/shape/input changes and JVP/VJP pass after.
  Complete source packaging regression first finds five missing benchmark files;
  copying/shipping benchmark harness resolves it. Combined contracts: 52/52 pass.
- [x] Scheduled reduction tracing regressions first reproduce two traces for
  successive inputs and independent output files. Stable module executor plus
  immutable field selection closes duplication; 87 output/continuation/observation
  contracts pass. Independent review finds arbitrary callable hashability and
  default bound-selector retention regressions; the weak-ownership fixes and
  complete review evidence are recorded below.
- [x] Fixed-source four-CPU lifecycle probe passes all 20 preset/custom cases in
  182.69 s with shared halo compilation and cold subsecond cache, compared with
  historical 331.55 s. This standalone measurement is not full-suite budget proof;
  it overlapped only lightweight regression checks/static work. Evidence:
  `test_logs/reorganization/lifecycle-halo-cache*`.
- Failed test approach: counting only advance traces missed fresh reducer wrappers,
  because checkpoint tracing was already cached. The maintained regression now
  records the real reducer trace and verifies changing numerical state/output.
- Failed integrated full attempt: JAX compilation aborts at
  `io/test_simulation_output.py::test_context_failure_closes_writer_and_final_partial_mean_is_discarded`
  after 1491.49 s, peak individual RSS 5436016 KiB. No complete correctness,
  coverage or JUnit result; original tested source hashes remained unchanged.
  The exact test passes alone in a fresh process. Abort cause is not yet proven.
- [x] Callback follow-up: weak identity keys preserve unhashable pure callbacks
  and release their captured resources; non-weak-referenceable callables use
  scoped partial owners and weak numerical proxies. RED reproduces all five callable failures and
  four lifetime failures; subclass selectors retain their existing behavior.
  Cache only exact ordinary functions with immutable default field selection and
  no history. A weak transition key owns each JIT wrapper, so expired transitions
  release compilation owners too. Avoid a global JIT that retains expired static
  callback keys. Combined real output/lifetime/reuse checks: 85/85 pass.
- Failed scoped-cache approach: forwarding arbitrary callbacks directly still
  exposed checkpoint's hashability requirement and let nested scan caches retain
  caller closures (four failing cases). Temporary strong partials plus weak
  numerical proxies during lowering close both gaps; their JIT owners and caller
  resources collect without clearing JAX caches. Default reusable transitions
  retain their weak-owned cache and existing trace-reuse assertions pass.
- [x] One coordinated pytest-xdist session now partitions worker CPU affinity
  before collection/backend initialization; no cases or scientific assertions
  are removed. Runtime regressions fail before the hook/option and pass after.
  Each worker owns a fresh cache inherited by its child probes. Shared unbounded
  worker disk cache rejected after installed JAX source review found unlocked,
  non-atomic writes. CI selects at most three workers, keeps module fixture reuse
  and fails immediately on worker crash. Full measured budget remains pending.
- Optional test-only dynamic-State AD drafts preserve all cases, literals,
  assertions and parameter-only derivative directions; measure sequentially if
  the current full run still exceeds 1200 s. No temporary JAX constant-handling
  flag, warm prior-session cache or reduced scientific selection is accepted.
- [x] First coordinated full run: 1392 passed, two expected CPU-only GPU skips,
  782.90 s external wall (779.48 s pytest), versus 2219.44 s previous full suite.
  All 1394 cases execute exactly once. Maintained coverage 4293/4536 (94.64%):
  3829 unchanged previously executed lines retain coverage, zero unchanged losses.
  Nine moved executable schema/halo lines are covered at their new owners; the
  remaining changed lines are canonical imports/docstrings or reviewed host
  callbacks. All ten scientific reference files retain their hashes.
- [x] Review's worker-test portability failures reproduced before fixing: one-CPU
  child placement and mocked absence of Linux affinity attributes. Adaptive child
  groups preserve both private cache lifecycles and real numerical assertions;
  portable mocks and explicit unsupported-platform errors cover placement.
  All eight runtime cases pass with taskset restricting the process to one CPU.
  No scientific case, tolerance, dtype or AD assertion is reduced. Optional AD
  drafts are rejected: modest/negative gains do not justify added test complexity.
- [x] Both final CUDA-only tests pass on the two Tesla P100s (15.99 s pytest).
  Fresh wheel/sdist content audit, strict Sphinx, clean installed literal public
  examples, lint/format/annotation/types and dependency checks pass.
- [x] Final reviewed cold correctness: 1393 passed, two expected GPU skips,
  783.11 s external wall (781.27 s pytest), all 1395 cases exactly once. Maintained
  coverage remains 4293/4536 (94.64%), zero lost unchanged lines. Three workers
  use disjoint 16-CPU groups on the 48-logical-CPU host, with independent fresh
  caches. All 221 source/test/benchmark Python hashes remain unchanged during
  verification. Final independent review has zero Critical/Important findings.
- [x] Final hierarchy: model, grid, physics (dynamics/transport/thermodynamics),
  numerics, execution, io and setups. All 38 previous paths retain typed module
  aliases; root API and old pickle imports remain compatible. Geometry/array
  schemas and shared halo trimming have explicit owners. The complete tree and
  old-to-new mapping are in docs/reference/package-hierarchy.rst; tests/README.md
  and tests/AUDIT.md record the reproducible full command and evidence. Maximum
  maintained module size: 786 lines. Commit the verified refactor on jax-only.
- [x] Structure, performance and documentation audits completed independently.
  Root retains all numerical equations; I/O no longer imports halo trimming from
  the parallel setup CLI. Canonical imports/testing/docs delivery remains underway.
- Continue in the user-designated `jax-only` checkout. Explicit implementation,
  autonomous development and full-suite-before-commit instructions govern this
  task; prepare a concrete tree/mapping/plan before production changes.

## 2026-10-12 — Local JAX CUDA 12 upgrade

- [x] Updated `.venv-latest` as requested to `jax[cuda12]==0.11.2`:
  JAX, jaxlib, CUDA 12 plugin and CUDA 12 PJRT all report 0.11.2.
  Dependency validation reports no broken requirements.
- [x] Both GPU-only tests pass again with CUDA explicitly required:
  two-GPU scheduled output and GPU CLI snapshot, zero failures/errors/skips,
  16.72 s external wall time. All 163 Python source hashes remain unchanged.
- Repository requirements pins remain unchanged; this is a local environment
  update. The full correctness suite under 0.11.2 was not run in this focused
  verification. Install report, before/after environment snapshots and GPU
  evidence: `test_logs/test-runtime/*jax-0.11.2*`.

## 2026-10-12 — GPU-only verification

- [x] With explicit user approval for native local GPU access, JAX 0.11.1 /
  CUDA 12 detects both Tesla P100-PCIE-16GB devices. Run explicitly requires
  CUDA and disables process-local preallocation for the parent/child probes.
- [x] Both previously CPU-skipped tests pass: two-GPU scheduled output and
  the GPU CLI finite snapshot. Two passes, zero failures/errors/skips in
  31.54 s external wall time. Evidence: `test_logs/test-runtime/gpu-only.*`
  and `gpu-only-resources.txt`.
- All 163 Python hashes still match the completed full CPU validation;
  no code, test, tolerance or environment dependency changes. Full GPU-suite
  and actual multi-host verification were not part of this focused run.

## 2026-10-12 — Unit-test runtime reduction (started 2026-10-11)

- [x] COMPLETE (@root): measure and reduce full correctness runtime, audit newer
  lifecycle tests for redundant coverage, and apply float32 to appropriate
  operational contracts. Independent coverage/performance/quality audits pass;
  root owns the single pytest process. Plan:
  `docs/superpowers/plans/2026-10-11-test-runtime.md`.
- [x] Fresh baseline on `jax-only`: 1283 passed, two expected GPU skips,
  2670.82 s pytest / 2681.20 s external wall time; peak process RSS 3910968 KiB.
  All 161 baseline Python hashes match; maintained coverage remains
  4058/4296 = 94.46%. Python 3.14 coverage's sysmon core warns that dynamic
  per-test contexts are incomplete; compare authoritative executed-line sets.
- Initial measured bottlenecks: restart 453 s, public lifecycle sharding 397 s,
  rollout setup/AD 414 s and closed-y tests 297 s. Repeated identical evolved
  corruption checkpoints and serial references across sharding layouts are
  confirmed reuse candidates. No numerical changes or weakened checks planned.
- [x] Integrated seven independently reviewed harness paths after baseline
  completion: remove 12 duplicate rows/cases, use six float32 operational restart
  cases plus two float64 samples, retain every corruption and sharding case,
  copy immutable module checkpoint templates and reuse identical serial oracles.
  Template teardown verifies pristine bytes. Scientific references, tolerances,
  gradients, full selection and coverage exclusions remain unchanged.
- [x] Private session compilation cache: tests first fail on missing helper,
  then pass 2/2 after implementation. Fresh children execute different inputs,
  validate NumPy oracles and require an actual executable-cache hit; caller
  settings and success/failure cleanup are checked. Default one-second threshold
  is preserved; each full session starts with an empty owned cache.
- [x] Final full CPU correctness: 1273 passed, the same two GPU skips,
  2210.64 s pytest / 2219.44 s external wall, 17.22% less wall time (461.76 s).
  All 55 maintained files have identical executed/missing/excluded line sets:
  4058/4296 = 94.46%. All 163 tested Python and nine scientific fixture hashes
  match; all 55 production Python files are unchanged. No tolerance, numerical
  archive, skip, selection or exclusion changes. Independent ownership audit
  estimates 186 fewer explicit physics timesteps inside surviving/removed cases.
- [x] Measured operational State storage: all 70 fields use 22400 bytes in
  float32 versus 44800 in float64 for the same 4x6 island grid, a 50% saving.
  Whole-process peak RSS increased from 3910968 to 4016260 KiB (+2.69%,
  3.73 to 3.83 GiB); reduced array storage does not establish lower compiler RSS.
  One run per variant cannot identify the cause. Cumulative root high-water
  marks locate the final increase in rollout/AD; they are not module-local usage.
  Unmeasured module cache clearing is not introduced as a claimed memory fix.
- [x] Root cache monitoring: 110 executable hits, 123.13 estimated compile
  seconds saved; peak session disk cache 82712449 bytes including child entries.
  These counters omit child cache events. Both sessions start without a warm
  persistent cache; optimized run adds lightweight monitoring. GNU time RSS is
  the largest individual process, not concurrent process-tree memory.
- [x] Maintained Ruff/format/annotations/ty, dependencies, diff and strict Sphinx
  pass. Fresh wheel/sdist builds preserve all 55 maintained production files and
  ship the new cache regressions/audit/plan. Clean installed public imports,
  coupled exact restart, observations, selected input, custom geometry,
  two-CPU sharding and pure JIT/grad/JVP/VJP examples pass. The final focused
  change is verified for the required jax-only commit. Evidence prefix
  `test_logs/test-runtime/`; GPU and actual multi-host hardware verification
  remain unavailable. Whole-process memory reduction remains unproven.

## 2026-10-11 — Centralized lifecycle: public documentation and reproducible delivery

- [x] COMPLETE (@root): final delivery after migration commit 09b4df7.
  Integrated independently reviewed primary coupled quickstart, complete custom
  ocean example, migration guide, setup gallery, AD/integration guidance, README
  and lifecycle architecture. Public examples execute literal published code.
- Packaging regression written first: the real isolated source build omits
  harness helpers, scientific reference artifacts and architecture metadata.
  Updated manifest includes complete tests/docs, immutable capture text,
  DESIGN/CHANGELOG and the parallel launcher. Actual maintained RED identifies
  49 omissions; GREEN passes after integration.
- CPU CI now runs strict documentation builds and clean installed public examples.
  A real isolated Sphinx probe reproduces missing Furo, then passes after installing
  docs requirements; independent scoped rereview approves the dependency fix.
- Scientific/API/restart and documentation reviews close all reported blockers.
  Maintained lint/format/annotations/types, strict Sphinx and fresh builds pass.
  All 225 checked source/harness/docs archive artifacts match exact bytes. Clean
  installed literal examples pass coupled exact restart, observations, selected
  input, custom ocean, two-CPU sharding and pure JIT/grad/JVP/VJP. Strict Sphinx
  also passes from the extracted source archive.
- Final full CPU correctness: 1283 passed, two expected GPU skips, zero failures/
  errors in 2648.46 s. Maintained coverage 4058/4296 = 94.46%; whole-package
  coverage 87.29%. All 161 Python source/test and nine scientific fixture hashes
  match. Evidence prefix `test_logs/lifecycle-delivery-*`; requirement evidence
  is recorded in the implementation plan. GPU and actual multi-host hardware
  verification remain unavailable; distributed/repartitioned restart is rejected.
- Delivered immutable Python/JSON definitions, runnable global-grid initialization,
  one-step pure advance, host Simulation with absolute forcing/time, shared CLI
  examples, scheduled I/O/selected input and exact complete serial restart.
  Numerical kernels/tolerances and existing public rollout behavior are preserved.
  Independent legacy archives verify 1615 stored arrays; pure differentiation and
  real two/four-CPU boundary/output/resource checks pass. All eight milestones
  are complete; the final focused delivery is ready for the required jax-only
  commit after metadata/archive close-out.

## 2026-10-11 — Centralized lifecycle: migrated examples and public sharding

- [x] COMPLETE (@root): task 7 after restart commit 99e6547. Maintained tests
  first reproduce missing regional/output helpers, five snapshot/collective
  contract failures and two zero-step CLI lifecycle failures. Integrated exactly
  ten independently reviewed source paths; focused and full verification pass.
- All presets and custom geometry/initial/constant or series forcing use cached
  addressable regional construction with global physical indices. Shared CLI
  execution retains explicit distributed bootstrap, calendar/output contracts,
  compatibility entry points and borrowed mesh ownership. Serial host work
  suspends ambient meshes and restores the caller's execution context.
- Independent original references cover 1540 arrays across seven trajectories
  and another 75 arrays for unequal legacy island timesteps. Every pure,
  compatibility and host path passes direct draft comparison. Four-CPU draft
  checks pass 20 run/output and 20 small-partition initialization cases; pure,
  rollout float64/float32 and closed-wall AD probes pass with frozen source hashes.
- Reviews reproduced unequal-timestep compatibility and collective finite-status
  defects before fixes; fresh rereviews approve. Ambient-mesh and nested-output
  regressions also pass direct RED/GREEN. A final independent scientific/JAX/
  restart audit finds no blocker; integrated full gates remain required.
- Initial maintained checks caught one import ordering issue and an inference
  error in immutable original capture code. Generators remain byte-for-byte
  archived source artifacts (`.py.txt`), with original hashes and reproduction
  instructions; current source/tests pass unchanged maintained type checks.
- Focused integrated verification: 40/40 pass in 126.54 s, plus 3/3 retained
  nonfinite-output regressions. Strict docs, maintained checks, fresh builds and
  clean installed literal coupled/restart/custom/sharded/pure-AD examples pass.
  Full CPU correctness: 1282 passed, two expected GPU skips, zero failures/errors
  in 2632.55 s. Maintained coverage 4058/4296 = 94.46%; whole-package coverage
  87.29%. All 159 Python and nine fixture hashes match. Evidence prefix
  `test_logs/lifecycle-migration-*`.
- GPU and actual multi-host verification remain unavailable. Complete source
  archive harness/metadata, primary quickstart and migration delivery remain.

## 2026-10-11 — Centralized lifecycle: complete serial restart

- [x] COMPLETE (@root): task 6 after output commit b5be7b6. Maintained tests first:
  missing restart_arrays collection error, then 36 missing-contract failures and
  26 fixture errors in 62 public/API cases before integrating seven reviewed
  source paths. Focused verification covered 159 contracts; three new fixture
  failures were corrected and those cases pass a fresh explicit rerun.
- Version 1 stores all native State fields and full halos, resolved definition,
  constants/configuration, typed input arrays and exact absolute clock/forcing
  position. Host validation precedes device allocation; atomic publication never
  overwrites. Serial CPU/single-GPU restoration uses the saved backend.
- Restored output starts in a new file at the original calendar/cadence; crossing
  mean windows are discarded and checkpoint instants are skipped, including zero.
  Independent review reproduced that zero-time replay before the isolated fix;
  its direct RED/GREEN and scoped rereview pass.
- Initial maintained checks identified two test-only typing/import issues; added
  an explicit geometry-presence assertion and import separation. Equations,
  source references and established tolerances are unchanged.
- The wall-traction fixture initially used a preset with a dry north row; use
  wet coupled-island geometry to exercise nonzero traction. Attempting to change
  a builtin's physics is deliberately rejected, so the test retains its coupled
  mode. Corrupt-variable fixtures now detach dimension-scale references before
  deletion; dangling references previously prevented construction of the intended
  dtype/shape defect. All original strong assertions remain.
- Maintained checks, Sphinx -E -W, fresh build and clean installed exact coupled
  save/resume quickstart pass. Full CPU correctness: 1239 passed, two expected
  GPU skips, zero failures/errors in 2157.86 s. Maintained coverage 4034/4251 =
  94.90%; whole-package coverage 87.62%. All 146 frozen Python source/test hashes
  match. Evidence prefix `test_logs/lifecycle-restart-*`.
- Migrated examples and broader public sharded initialization remain isolated.
  Seven immutable original trajectories, four-CPU initialization/output, both
  rollout precisions and pure/wall AD probes pass there; maintained integration
  and full verification remain required. GPU and actual multi-host checks remain
  unavailable.

## 2026-10-11 — Centralized lifecycle: scheduled output and selected input

- [x] COMPLETE (@root): task 5 after Simulation commit fff5356. Tests first:
  76 missing-contract failures out of 79 new cases before source integration.
- Reuse the existing scheduled scan, event planner and netCDF writer with
  absolute run offsets, dynamic forcing inputs and retained unfinished means.
  Requested observations share the same trajectory; default output retains no
  timestep history. Snapshots and selected serial input use established storage,
  physical validation, precision, dry/blocked-face masks and halo policies.
- Independent review reproduced and corrected the scheduled pair-carry selector
  before integration. Focused run: 269 passing, one expected GPU skip and one
  new rational-mean oracle failure, differing by one float64 ULP from host
  division. The assertion now allows that one ULP; numerical code is unchanged,
  exact sample counts/times/bounds remain asserted, and the focused case passes.
- Clean installed documented coupled split-output, optional observations,
  snapshot and selected-input smoke passes. Full CPU correctness: 1174 passed,
  two expected GPU skips, zero failures/errors in 1623.73 s. Maintained coverage
  3714/3870 = 95.97%; whole-package coverage 87.95%. All 141 frozen Python
  source/test hashes match. Maintained lint/format/annotations/types and diff
  checks, Sphinx -E -W and fresh wheel/sdist checks pass. Evidence prefix
  `test_logs/lifecycle-output-*`; GPU remains unavailable.
- Complete restart remains isolated. Review reproduced step-zero checkpoint
  instant replay; its isolated fix has independent rereview and direct runtime
  evidence. Complete restart integration/full verification remains pending.
  Public sharded initialization, migrated examples and final delivery remain.

## 2026-10-11 — Centralized lifecycle: host simulation milestone

- [x] COMPLETE (@root): task 4 after timestep commit e916eef. Fresh implementation
  and independent spec/quality review prepared isolated drafts while the previous
  full suite ran; root integrated tests first and owns all runtime/commit gates.
- [x] All 27 focused tests RED on missing public Simulation before integration,
  then 27/27 GREEN in 62.58 s. The facade reuses initialize_experiment, advance
  and unchanged veris.step with a separate (State, Diagnostics) carry. Constants
  are bound once; only absolute-position forcing-series slices are dynamic inputs.
- [x] Public RunResult reports final arrays, final diagnostics, per-call steps,
  absolute completed steps/model time, optional post-step observations and timings.
  Split runs equal uninterrupted legacy trajectories; unobserved zero runs retain
  identities without tracing, and observed zero runs work at forcing exhaustion.
  Exact clock and forcing preflight errors leave the simulation usable; tracing
  or synchronized execution failures close it before committing State/time.
- [x] Host guard, requested CPU allocation, context/idempotent cleanup, custom
  coupled geometry/masks and selected observations pass. Direct two-CPU public
  dynamics smoke compares every State/Diagnostics physical field with the legacy
  trajectory in both (2,1)/(1,2) closed-y orientations, including successive runs
  and actual observations. Mesh resources remain explicit and caller-owned.
- [x] Independent review found no blocking static issue. Maintained lint, format,
  annotations/types and diff checks pass; Sphinx -E -W and fresh wheel/sdist build
  pass. Clean installed documented split-run and observation smoke passes.
- [x] Full CPU correctness: 1095 passed, two expected GPU-only skips, zero
  failures/errors in 1509.30 s. Maintained coverage 3477/3624 = 95.94%; whole
  package coverage 87.43%. All 136 source/test hashes match the full-run tree.
  Final maintained lint/format/annotations/types and diff checks pass. Evidence
  prefix `test_logs/lifecycle-simulation-*`. GPU remains unavailable.
- Scheduled output continuation, selected input, complete restart, broader
  sharded initialization, migrated examples and final quickstart/migration guide
  remain required. Task5 is being prepared only as an isolated source overlay.

## 2026-10-11 — Centralized lifecycle: shared pure timestep milestone

- [x] COMPLETE (@root): task 3 after initialization commit fff4cd4. Root owns
  implementation and the single pytest runner; execution audit prepared independent
  legacy all-field, forcing-boundary and differentiation tests before implementation.
- Shared advance performs one physical timestep; existing veris.step retains its
  multi-step rollout contract. Reuse dynamics_transport and Growth in reference
  order, refreshing only supplied forcing before numerical stages.
- Verified all three modes, both precisions/boundaries, prescribed
  versus recursive heat, ocean normal-face forcing, JIT and initial/heat/wind
  gradients, with full correctness and maintained/docs/build/install gates.
- [x] Original 29 missing-API cases RED, then GREEN. Review reproduced calm
  wind's spurious 0.5 derivatives, unchecked State storage/precision and disabled
  x64 truncation. Stable hypot forward plus analytic JVP now selects the existing
  zero norm linearization and avoids large-component derivative overflow.
  Expanded focused 35/35 pass, including all-field preset/forcing and AD checks.
- Packaging audit discovered existing sdist omissions of conftest, probe/helper
  modules and reference fixtures. Task8 must add an explicit source-test manifest
  with a reproducing archive test; wheel runtime modules already match source.
- Clean installed smoke reproduced a documentation binding error: a partial
  with keyword-bound settings cannot accept the rollout's second positional
  forcing argument. The dynamic-forcing example now uses an explicit binary
  transition calling advance(state, settings, constants, prescribed).
- Direct two-CPU forcing probe reproduced an explicit-sharding AD failure:
  spatial fields formed from replicated parameters retained replicated types,
  incompatible with shard_map's physical partition specs. Pure advance now
  explicitly reshards prescribed inputs before local padding/exchange; the
  test checks both orientations, every field/diagnostic and forcing derivatives.
- [x] Direct two-CPU probe passes in 114.23 s: eager/JIT all-field and diagnostic
  equivalence, local spatial forcing halos, explicit speed and nonzero wind/ocean
  grad/JVP/VJP/finite differences on (2,1)/(1,2) meshes, including closed-y
  two-cell partitions. Final read-only scientific/API review found no blocker.
- [x] Full CPU correctness: 1068 passed, two expected GPU-only skips, zero
  failures/errors in 1458.82 s. Maintained coverage 3325/3472 = 95.77%; whole
  package 3325/3825 = 86.93%. All 134 source/test hashes match the full-run tree.
  Maintained lint/format/annotations/types and diff checks pass; Sphinx -E -W,
  fresh wheel/sdist runtime byte comparison and clean installed public timestep,
  rollout and documented heat-gradient smoke pass. Evidence prefix
  `test_logs/lifecycle-physics-*`; GPU remains unavailable.
- Next: commit the passing timestep milestone on jax-only, then integrate and
  verify the isolated Simulation draft. Output/restart/sharded initialization,
  example migration and final documentation remain required; the goal is active.

## 2026-10-11 — Centralized lifecycle: declarative initialization milestone

- IN PROGRESS (@root): task 2 of the full lifecycle plan, following the passing
  definition commit 949546d on jax-only. Root schedules the single pytest runner;
  initialization/execution/output agents provide focused implementation or review.
- [x] Tests first: declarative and initialization APIs missing (49 RED cases),
  preset controls missing (8 RED cases). Added grouped versioned JSON, deep
  overrides, exact dates/durations and resolved metadata; builtin definitions use
  maintained scenario registries and shared numerical defaults.
- [x] Added public OceanGeometry and initialize_experiment. Custom geometry uses
  physical global cells; the existing adapter supplies metric equations, then
  initial/forcing fields, masks, derived quantities and field-aware halos are
  prepared. Preset all-field initialization oracles passed in both island
  precisions and all three physics modes; growth scenarios are now configurable.
- Review regressions reproduced before fixes: coastal velocity masking, derived
  float32 load/inverse overflow, x64 rejection before allocation, backend/mesh
  mismatch, negative Kelvin growth ocean temperature, ndarray.resize metadata
  mutation, minimal preset-file defaults, Decimal-context duration rounding and
  schema version relabeling. Also replaced a conservative four-times-maximum
  area bound with actual neighboring sums after a RED valid-input regression.
- [x] Focused final suite: 120/120 pass. Maintained lint, format, annotations and
  types pass. Sphinx -E -W passes; fresh wheel/sdist contain current modules and
  tests verbatim. Clean wheel installation outside the source path verifies JSON,
  preset and custom initialization using only public imports, with float32 arrays.
- [x] Full CPU correctness: 1032 passed, two expected GPU-only skips, zero
  failures/errors in 1255.08 s. Maintained coverage 3231/3361 = 96.13%; whole
  package 3231/3714 = 87.00%. All 130 source/test hashes match the full-run tree;
  final maintained lint/format/annotations/types and diff checks pass. Evidence
  prefix: `test_logs/lifecycle-initialization-*`. No redundant fast run before
  the planned full suite. GPU verification remains unavailable on this host.
- Next: focused passing initialization commit. Shared
  pure timestep, Simulation/time, output continuation, restart and CLI migration
  remain required; drafts for pure physics and output tests are outside collection.

## 2026-10-11 — Centralized lifecycle: definition milestone

- IN PROGRESS (@root): implement the complete new AGENTS.md lifecycle Goal.
  Design: `docs/superpowers/specs/2026-10-11-model-lifecycle-design.md`;
  full implementation plan: `docs/superpowers/plans/2026-10-11-model-lifecycle.md`.
  Work remains on `jax-only` (baseline 8818ea5); GitHub confirms that the maintained
  branch exists. Do not use main. No code commit before full correctness passes.
- [x] Read DESIGN, progress history, public interfaces, initialization, setup
  physics, rollout/output, storage/calendar, packaging, docs and tests. Specialized
  read-only initialization, execution and output audits completed; root schedules
  all pytest. Key contracts: keep legacy builtin initial intermediates exact,
  preserve island prescribed versus growth recursive heat fluxes, adapt global
  dimensions into local packed halos, and add real schedule/time continuation.
- [x] Added public immutable Grid, Execution, Forcing and Experiment. Numerical
  and physical defaults/validation stay with Configuration/PhysicalConstants.
  Definitions validate global/local/storage dimensions, prescribed field names,
  scalar/spatial/time shapes, concentration/thickness/temperature, series lengths,
  precision inheritance, active timestep and coupled equal-timestep requirement.
  Constructors allocate no model arrays and select no backend.
- [x] RED before implementation: all 51 original new contract cases failed on
  missing public APIs. Focused final suite: 55/55 pass, including import/backend
  purity and regression tests for geometry aliasing, huge-clock overflow and
  NumPy shape/dtype mutation. Defensively copied arrays own immutable bytes and
  also protect metadata. Existing veris.step behavior and kernels unchanged.
- [x] Independent review reproduced NumPy metadata mutation despite write=False;
  fixed with failing tests covering forcing, initial conditions and geometry.
  Prior geometry aliasing and clock failures were also reproduced before fixes.
  Float32 field representability remains a required initialization-boundary check
  in task 2 (overflow and positive-kelvin/metric underflow before allocation).
- [x] Baseline fast suite passes (92 selected cases). Maintained Ruff, format,
  annotations and ty checks pass. Fresh wheel build and clean target installation
  outside the source import path pass public API/validation smoke checks.
  Sphinx -E -W succeeds and the rendered definition page is present.
- [x] Full CPU correctness: 967 passed, two expected GPU-only skips, zero
  errors/failures in 1251.35 s. Maintained coverage 2839/2927 = 96.99%; whole
  package 2839/3280 = 86.55%. All source/test hashes match the full-run snapshot.
  Fresh final Ruff/format/annotations/ty and diff checks pass. Evidence:
  `test_logs/lifecycle-definition-full.log`, full.xml, coverage.json and
  source-hashes.json. Focused RED/GREEN logs and XML use the same prefix;
  docs/build/install artifacts are also under `test_logs/lifecycle-definition-*`.
  One explanatory test lint comment changed during the full run; executable AST
  was checked identical and final hashes refreshed. No numerical source changed.
- Environment: activate existing `.venv-latest` (Python 3.14.7/JAX 0.11.1).
  Automatic backend probe reported CUDA_ERROR_NO_DEVICE and CPU; all checks now
  set JAX_PLATFORMS=cpu. GPU validation is unavailable on this host so far.
- Next: focused passing definition commit, then task 2 JSON
  definitions/presets/runnable custom initialization. Simulation facade, shared
  timestep, time/output continuation, complete restart, migrated CLIs, primary
  quickstart and migration guide are NOT implemented yet. The goal remains active.

## 2026-10-07 — Portable benchmark CPU metadata

- [x] Reproduced the Darwin AttributeError in two regression cases before the
  fix; the Linux sparse-mask test also exposed the missing CPU-count metadata.
- [x] Feature-detect `os.sched_getaffinity`: Linux keeps sorted allowed CPU IDs;
  platforms without the API report `cpu_affinity=null` and use `os.cpu_count()`
  for separate logical-CPU metadata (including its unknown/null case).
- [x] Updated the real CLI/trace test to assert platform-appropriate metadata;
  documented the fallback and restricted taskset instructions to Linux.
- [x] Native Darwin benchmark tests: 21/21 passed, including real separate
  XPlane/Perfetto traces. CI-scope Ruff, formatting, annotations, ty and diff
  checks pass. Independent read-only review found no correctness blocker.
- [x] Sphinx clean warnings-as-errors build passes; rendered index title and
  thesis link verified. Initial sandbox build could not fetch the Python
  inventory; authorized network rebuild passed. Package name/version query works.
- [x] Full CPU suite: 911 passed, two expected GPU-only skips, one sandbox
  localhost-bind permission failure. The affected two-process reduction/AD test
  passed with socket access (1/1), yielding 912 passing tests overall. Maintained
  coverage: 2654/2732 = 97.14%; whole-package coverage: 2654/3085 = 86.03%.
  No numerical tolerances changed. No pytest remains running.
- [x] Ready to commit on jax-only, including the user's existing setup.py and
  docs/index.rst edits. Evidence: /private/tmp/veris-macos-{profile-tests,
  full-tests,reduction-tests,docs}.log and veris-macos-coverage.json.
- [x] Git author identity is now configured as Roman Nuterman
  <nuterman@nbi.ku.dk>; the earlier missing-identity blocker is resolved.
  User authorized committing and pushing these verified changes to jax-only
  with the Git CLI. Sandbox index-lock denial requires authorized escalation.
- Environment: created project-local .venv because no existing venv was present.
  Sandboxed dependency download failed on DNS; authorized network retry worked.

## 2026-09-17 — Core dynamics ownership and test redundancy audit

- [x] Moved setups/_physics.py to veris/dynamics.py, preserving the
  dynamics_transport interface and numerical ordering. Root handled the move,
  setup/integration tests, audit documentation and all pytest scheduling.
- Audit ownership: numerical_audit reviews numerical/gradient/boundary tests;
  output_audit reviews IO/output/calendar/restart tests; contract_audit reviews
  configuration/initialization/typing/precision tests. No concurrent pytest.
- [x] Moved core dynamics and its existing sequence test; both callers import
  veris.dynamics. New-path test failed first on ModuleNotFoundError, then all 12
  focused dynamics/compiled tests passed. Executable stage AST is unchanged.
- [x] Reviewed all 66 modules; consolidated 18 collected cases (929 -> 911) and
  repeated calls/assertions. Full removed-to-retained mapping is in tests/AUDIT.md.
  Independent review caught weaker corner-mask equality; restored exact dry/wet
  assertions inside the packed oracle before full verification.
- [x] Maintained Ruff, format and ty checks pass. Removing the rollout forwarding
  helper exposed an inferred keyword-dict type; explicit dict[str, Any] preserves
  intentional invalid-input test cases. No production typing change needed.
- [x] Clean wheel contains the new module and both callers verbatim, excludes
  the old private module. Initial wheel reused stale build/lib content; use a
  fresh --build-base when checking deleted/renamed modules. Sphinx -E -W passes
  after allowing its configured remote inventory download; no docs warnings.
- [x] Full CPU suite: 909 passed, 2 expected GPU-only skips in 1251.81 s.
  Maintained coverage 2654/2732 = 97.14%; whole-package 2654/3085 = 86.03%.
  Every covered/missing/excluded line set is identical to baseline after mapping
  the relocated module. Source/test hashes match the full-run snapshot.
- [x] CUDA: 120 selected tests passed, no failures/skips in 199.83 s, including
  both GPU-only cases skipped by CPU. Final review confirms all 66 modules are
  accounted for and no additional whole-test duplicate is safely removable.
- [x] Completed verification for the core move and test compaction on jax-only.
  Evidence: test_logs/dynamics-audit/ (CPU/GPU XML, coverage comparison, module
  pass counts, source hashes, clean wheel and Sphinx HTML). Runtime artifacts
  remain untracked. Tests use .venv-latest, not nonexistent .venv.
- Baseline: jax-only bc0864f; 929 cases, 927 CPU passes and two GPU-only skips;
  maintained coverage 2654/2732 (97.14%) from the preceding full run.
- Remove repeated contracts only with explicit retained owners. Independent
  equation, branch, AD, precision and distributed checks remain meaningful even
  when they execute common kernels. Record decisions in tests/AUDIT.md.

## 2026-09-17 — Production simplification

- [x] Audited production duplication from published `jax-only` commit `041df78`.
  Shared directional advection limiter stencils and sweep updates; preserved
  public JIT wrappers, arithmetic order, zero-slope AD guards and directional
  wall policies. Advection shrank by 27 lines.
- [x] Shared the identical local momentum/transport sequence between island
  and run_dyn in setups/_physics.py. Cooling reset, Growth and final halo
  refresh remain in the callers at their original positions. No State fields,
  settings, physical constants or metadata registries changed.
- [x] Added eight cases: explicit scalar-index flux oracles for both axes/wall
  modes with signed flow, nonuniform metrics, fractional masks and capped/zero
  slopes; weighted JVP/VJP/finite differences; exact stage sequence and stale
  forcing-halo checks. Stage tests failed on the missing module before extraction.
  All 333 original test bodies/parameters and 113 public function signatures and
  decorators are unchanged. No tolerances relaxed or existing tests removed.
- [x] Original-source baseline: 48 checks plus six new flux/gradient cases pass.
  Post-refactor focused checks: 31 transport and 12 stage/driver cases pass;
  fast suite: 93 passed. All 72 old/new primal/JVP/VJP comparisons are bitwise
  equal across float32/64, wall modes, extensive/intensive transport and flux APIs.
- [x] Final full CPU correctness: 927 passed, two expected GPU-only skips in
  1253.17 s. Maintained coverage: 2654/2732 = 97.14%; whole package: 86.03%.
  The 80% gate passes. All 78 previously uncovered maintained statements are
  unchanged; refactoring removed 14 redundant covered statements. Advection
  and the new shared stage have 100% coverage.
- [x] CUDA: 97 selected tests passed, zero failures/skips, in 860.02 s, including
  both GPU-only cases skipped on CPU. Two duplicate four-CPU cases were
  deselected only in this CUDA run; both passed in the full CPU suite. Separate
  two-GPU probe passed both grid orientations, two local y sizes, all four halo
  policies, both slip modes, two coupled steps, all State fields and JVP/VJP/FD.
- [x] Final maintained Ruff, format, annotation and ty checks pass. Wheel build
  succeeds and contains all four changed production modules verbatim. Final
  source/test hashes match the verified snapshot; all verification processes
  completed. Integration target is `jax-only`; runtime artifacts stay untracked.
- Independent transport and setup-stage reviews found no blocker. Keep distinct
  IO precision/execution paths, wall ordering and cooling shard specifications.
  Retain EVP carry: existing target-sensitive fusion benchmarks do not justify
  changing its JAX/AD staging as part of this maintenance refactor.
- Evidence: test_logs/simplification-{full,gpu,stage,fast}.{log,xml},
  simplification-gpu-closed.log, simplification-equivalence.{py,log},
  simplification-{coverage,summary,contract-audit,source-hashes}.json and
  simplification-build.log. Initial development command referenced nonexistent
  test_transport_equations.py, collected nothing, and was corrected to actual
  transport/AD/boundary owners before implementation.

## 2026-09-16 — Precision-focused test compaction

- IN PROGRESS: assign precision by contract, per user request. Validation,
  metadata, shapes, masks and indexing run once; float64 owns precise equation
  and derivative references; float32 owns operational/stability checks. Both
  remain only for dtype/conversion, sensitive thresholds and a small integration
  sample. Production code and numerical tolerances remain unchanged.
- Baseline: local jax-only commit 62749ef, 950 collected cases, 948 CPU passes
  plus two GPU-only skips; maintained coverage 2668/2746 (97.16%). Existing
  evidence: test_logs/test-compaction/. GitHub does not yet resolve this local
  commit; it is not evidence of a published change.
- Task ownership: root handles rollout/restart, documentation and all pytest
  scheduling; precision_numerics handles numerical precision sweeps;
  precision_schema handles validation/metadata mixed into dtype tests.
- [x] Collection reduced 950 → 921 (29 cases); additional reduction inside
  tests removes repeated metadata/validation and float32 analytic/FD work.
  Exact ownership and retained both-precision contracts: tests/AUDIT.md.
- [x] Fast suite: 93 passed, 828 deselected in 90.96 seconds. Maintained Ruff,
  formatting, annotations and ty checks pass. Independent scientific review
  found no coverage blocker or over-removal. Production hashes exactly match
  the previously verified baseline.
- [x] Full CPU correctness: 919 passed, two expected GPU-only skips in
  1241.78 seconds (20m42s), versus 1654.90 seconds (27m35s) before this pass:
  413.12 seconds shorter (25.0%). This is end-to-end test timing, not a solver
  benchmark. Native local socket access enabled the distributed JAX tests.
- [x] Maintained coverage remains 2668/2746 = 97.16%; whole-package coverage
  remains 86.09%. Exact executed-line sets are unchanged in every maintained
  source file. Tested source/test hashes match; the 80% gate passes. Evidence:
  test_logs/precision-compaction/{full.log,full.xml,coverage.json,comparison.json}.
- [x] Unsandboxed CUDA verification: 192 passed, no skips/errors/failures,
  including both GPU-only cases skipped on CPU, in 788.50 seconds.
  CUDA was explicitly required and both Tesla P100 GPUs were detected. The
  selected modules also retain their explicit CPU subprocess checks. Evidence:
  test_logs/precision-compaction/{gpu.log,gpu.xml,gpu-summary.json}.
- [x] Independent scientific and documentation reviews found no blocker.
  All test processes completed; final source/test hashes still match. User
  requested committing and pushing the verified changes to remote jax-only.
  Fetched remote 4246f23; local history is a fast-forward extension.

## 2026-09-16 — Test suite compaction

- [x] Audited and compacted repeated assertions/trajectories; root owned
  setup/integration and pytest scheduling, audit_physics reviewed numerical tests,
  audit_types reviewed schema/initialization, audit_output reviewed output/storage.
- Baseline HEAD 4246f23 matches all recorded source/test hashes from the passing
  1057-test CPU run (two GPU-only skips), 1951.70 seconds, maintained coverage
  2668/2746 = 97.16%. GitHub confirms the published commit.
- Plan: `docs/superpowers/plans/2026-09-16-test-compaction.md`. Preserve unique
  branches and independent oracles; measure repeated actions as well as cases.
- [x] Reviewed all 66 original test modules. Collection reduced 1059 → 950
  (109 cases, 10.3%); 65 test modules remain. Removal-to-retained mappings
  are in `tests/AUDIT.md`. No production changes or relaxed tolerances.
- [x] Fast development suite: 95 passed, 855 deselected in 65.73 seconds.
  Maintained Ruff, format, annotation and ty checks pass. Independent review
  preserved exact shape checks in the rollout comparison helper and corrected
  the ocean duplicate explanation to acknowledge float32 metric rounding.
- [x] Full CPU correctness: 948 passed, two expected GPU-only skips in
  1654.90 seconds (27m35s), versus recorded baseline 1951.70 seconds (32m32s).
  This measures end-to-end tests; solver performance remains in its benchmark.
- [x] Maintained coverage unchanged at 2668/2746 = 97.16%; exact executed-line
  sets match the baseline for every maintained source file. Whole-package
  coverage remains 86.09%. All tested source/test hashes still match; 80% gate,
  final Ruff/format/annotation/ty and diff checks pass. Evidence in
  `test_logs/test-compaction/{full.log,full.xml,coverage.json,comparison.json}`.
- [x] Focused CUDA verification: 211 passed, no skips/errors/failures, in
  211.57 seconds, including both GPU-only cases omitted by the CPU run.
  Evidence: `test_logs/test-compaction/{gpu.log,gpu.xml,gpu-summary.json}`.
- [x] Independent numerical and test-ownership reviews found no blocker. Final
  hash/coverage comparison remains exact; all test processes finished. The
  verified compaction is committed back to `jax-only`; local runtime artifacts
  remain untracked.

## 2026-09-16 — Scheduled output reductions

- User approved device-side mean sums/counts, schedule-driven chunks and
  compilation without duplicate warmup execution. Spec and plan:
  `docs/superpowers/specs/2026-09-16-scheduled-output-design.md` and
  `docs/superpowers/plans/2026-09-16-scheduled-output.md`.
- Branch `scheduled-output-reductions` starts from verified/pushed `jax-only`
  aba67ef. Root owns implementation orchestration and all pytest processes;
  schedule_audit independently checked sampling/calendar/collector invariants.
- Preserve float64 means, half-open windows, complete-window filtering and
  initial-sample behavior. Existing arbitrary callbacks retain bounded histories;
  maintained OutputManager callbacks select the new reduced scan path.
- Environment: all Python/tests/checks use `.venv-latest`. Initial precision
  probe tried the removed `jax.experimental.enable_x64`; installed JAX exposes
  the scoped context at `jax.enable_x64` instead.

- [x] Compilation-only preparation: runtime regression reproduced duplicate
  transitions in the old warmup. Lower/compile now precedes the single actual
  trajectory, including generic observed chunks and no-output full scans.
- [x] Exact calendar planner and exclusive reduced writer: all 42 focused cases
  pass after observed missing-implementation failures. Device sums/counts replace
  histories for scheduled callbacks, with event-aligned chunks and optional cap.
- [x] All 27 initial output oracle cases pass; JAX scan structure confirms no
  history or timestep input vector, and constant spatial storage for 10/1000 steps.
  Four-device CPU output matches serial records, with eight record-only gathers.
- [x] Review regressions reproduced missing checkpoint/callable validation,
  complex/Boolean means silently cast to real, and late manager freshness checks.
  Host preflight now rejects these before physics tracing or output effects.
- [x] Real two-step growth final State is bitwise equal with/without scheduled
  output in both precisions; direct half-open mean records agree. Float64 sums
  preserve float32 cancellation and the caller's original x64 tracing context.
- Validation notes: installed JAX scan metadata uses FTTuple.unpack(), not tuple
  indexing. GPU alias tried unavailable ROCm; explicit CUDA selection works.
  First CUDA run passed 29 cases; new sharded GPU test was skipped by its overly
  narrow environment marker, now corrected to recognize comma-separated backends.
- Maintained Ruff/format/annotation/ty and Sphinx warnings-as-errors pass.
- [x] Required two-GPU output reduction and real-physics checks passed (three
  tests, no skips). Two-process CPU probe also passed: non-addressable State/sum
  arrays, replicated counts, exactly five record collections and no nonwriter file.
  Probe logs: `test_logs/scheduled-gpu-sharded.log` and
  `test_logs/scheduled-multihost.log`.
- [x] Full CPU correctness: 1057 passed, 2 GPU-only skips, no failures/errors in
  1951.70 s. Maintained coverage: 2668/2746 = 97.16%; whole-package coverage
  86.09%. The 80% gate passes without additional exclusions.
- [x] All 150 tested source/test/documentation hashes match the final files.
  Evidence: `test_logs/scheduled-full-cpu.log`, `scheduled-full-results.xml`,
  `scheduled-coverage.json` and `scheduled-source-hashes.json`.
- [x] Final-source GPU CLI recheck passed without skips after explicitly
  confirming the GPU backend. Both GPU-only cases skipped by the CPU suite have
  passed on CUDA; logs include `test_logs/scheduled-final-gpu.log` and XML.
- Independent numerical, test-quality and documentation review is complete;
  reproduced review findings were fixed before the full run.

## 2026-09-16 — Optional closed global y boundaries

- Root implements `enable_cyclic_y=True` registry/configuration option on branch
  `optional-closed-y`, based on `jax-only`; boundary_review independently audits
  C-grid wall indexing and writes distributed value/AD probes. Root owns pytest.
- Closed mode keeps x periodic and exchanges internal y partitions, with dry
  global exterior masks and impermeable normal faces. Generic scalar halos use
  nearest-edge extension to retain valid temperatures and metrics; masks, ice
  amounts, velocities and meridional fluxes receive explicit wall treatment.
- Reproduced missing configuration in four serial tests before implementation.
  Coupled tests then reproduced island setup overwriting closed masks;
  initialization and whole-step field-aware refreshes now address that path.
- First transport FD fixture had constant slopes at the Superbee kink; changed
  to unequal quadratic slopes to test smooth derivatives, without relaxing
  tolerances. Conservation and its JVP/VJP passed before this fixture change.
- Environment: `.venv` absent; using established `.venv-latest` as instructed.
  Full regression, distributed/AD checks and documentation are complete below.
- [x] Serial focused checks pass: two-step coupled JVP/VJP/FD with both slip
  settings; mass and intensive-field conservation; closed setup/allocation.
- [x] Independent distributed halo oracles exposed a two-cell partition edge
  case: close the owned south normal face before sending it to a neighbor.
  After correction, all six layout/size combinations pass values and JVP/VJP.
- [x] Numerical review exposed generic refresh overwriting physical north-wall
  shear. Four regressions reproduced it; dedicated shear policy now retains
  no-slip traction and zeros free-slip wall stress. Thirteen focused serial
  boundary/stress tests pass. Distributed shear coverage added for final suite.
- Sphinx initial build encountered sandbox DNS failure fetching Python's
  intersphinx inventory; network-enabled retry passed. No content warning found.
- [x] Final Sphinx `-E -W` build passes with network inventory access. Fresh
  independent review found no numerical blocker; documented initialization-time
  topology changes and selected-field snapshot limits for northern wall shear.
- [x] All six four-CPU halo cases pass edge/zero/normal/shear value and independent
  JVP/VJP oracles. Two-CPU float32 coupled runs pass both slip modes, two steps,
  all-field serial agreement, wall shear and initial-state/forcing JVP/VJP/FD.
  Four-CPU float64 coupled tests also pass both slip modes in the full suite.
- [x] Full CPU suite: 958 passed, one GPU-only skip, one sandbox socket-permission
  failure in the existing two-process reduction test (1898.27 s). That test
  passed with socket access on rerun, giving 959 passing tests overall. No
  source change, tolerance change or test exclusion was needed for the rerun.
- [x] Maintained coverage 2450/2535 = 96.65%; whole package 2450/2888 = 84.83%.
  Maintained Ruff, formatting, annotation and ty checks pass; 80% coverage gate
  passes. All 109 source/test hashes still match full-suite inputs.
- Evidence: `test_logs/closed-y-full.log`, `closed-y-results.xml`,
  `closed-y-coverage.json`, `closed-y-reduction-rerun.log`, `closed-y-float32.log`,
  `closed-y-shear-halos.log`, `closed-y-docs.log`, and `closed-y-test-inputs.json`.
  Development reproductions are retained in `test_logs/closed-y-development/`.
- [x] Follow-up GPU validation: the CPU-skipped
  `test_gpu_cli_writes_finite_output` passed on the two local Tesla P100 GPUs
  using `.venv-latest` and an explicitly required GPU backend (no CPU fallback).
  Its two-step parallel CLI output has finite 12-by-16 physical fields. XML
  confirms one pass, zero skips/errors/failures. All 109 source/test hashes still
  match the CPU suite, giving 960 passing tests across CPU/GPU verification.
  Evidence: `test_logs/closed-y-gpu-test.log` and `closed-y-gpu-results.xml`.

## 2026-09-15 — Universal scan step and checkpointed AD

- [x] Implemented public `veris.step(initial, advance, steps, ...)` with lax.scan,
  default-enabled JAX checkpointing, optional time-indexed forcing and selected
  observation histories. Auxiliary diagnostics stay outside the State carry.
  Static count/forcing validation and zero-step behavior are documented/tested.
- [x] Shared host runner migrates growth, dynamics and parallel CLIs to scans.
  History output uses at most eight samples per chunk, selecting only requested
  fields; disabled histories allocate no observations. Warmup covers full/tail
  shapes without advancing the requested trajectory. Sampling stays host-only.
- [x] Fixed a newly exposed float32 JVP overflow in solve4temp with equivalent
  vapor-pressure algebra. Reproduced first in a dedicated public solver test,
  then verified using JVP/VJP/FD and an independent heat-balance derivative.
  Repair committed separately as a8b4361.
- Approved design and implementation plan: docs/superpowers/specs/2026-09-15-universal-step-design.md
  and docs/superpowers/plans/2026-09-15-universal-step.md. Root coordinated pytest;
  specialized agents implemented/reviewed setup validation, host I/O and docs.

### Final verification
- [x] Full CPU suite: 943 passed, one GPU-only skip, zero failures/errors in
  1520.53 s. Maintained coverage 2406/2482 = 96.94%; whole-package coverage
  84.87%. Full-suite XML and coverage: test_logs/scan-full-results.xml and
  test_logs/scan-full-coverage.json. New distributed AD cases account for much
  of the added runtime; ordinary development should retain --fast selection.
- [x] All 18 GPU regression assertions passed: all four setup types in both
  precisions, explicit-step field comparisons, checkpoint on/off JVP/VJP/FD,
  recursive growth heat-flux behavior, and the GPU-only CLI test skipped on CPU.
  These use the actual pytest test functions in a standalone GPU harness while
  only one CPU pytest instance runs. Evidence: test_logs/scan-gpu-cases.json.
- [x] Actual two-P100 sharded dynamics and coupled probes pass both precisions;
  actual four-CPU-device forward/AD probes pass both precisions in the full suite.
- [x] Growth, dynamics and two-GPU parallel CLIs each ran nine timesteps with
  netCDF histories, crossing an eight-step chunk and a one-step tail. All ten
  sample times, finite physical shapes and final/history equality verified.
  Evidence: test_logs/scan-gpu-cli-validation.log and scan-gpu-*.nc.
- [x] Provided SLURM launcher job 65267175 completed on node453 with exit0:0.
  Two-process nine-step output and history match serial explicit steps within
  5.56e-17 for hIceMean/uIce; all timestamps verified. Evidence:
  test_logs/scan-multiprocess-validation.json and scan-multiprocess-65267175.log.
- [x] Independent implementation and numerical-fix reviews found no blocker.
  Sphinx -E -W, maintained Ruff/format/annotation/ty and 80% coverage gate pass.
  All 121 source/test/document hashes match the full-suite input snapshot.
  No forward tolerances relaxed; no physics thresholds changed. Validation is
  complete; local artifacts remain under untracked test_logs/.

### Failed approaches resolved

- Initial setup run: 35 passed, 3 failed. Float32 coupled/growth/ocean JVPs were
  NaN with finite VJPs; the same defect reproduced in explicit Growth without
  scan. The inverse-vapor-power humidity derivative caused intermediate overflow;
  no tolerances or fixtures were relaxed to hide it. A pre-repair sharded probe
  reproduced the same defect; all post-repair checks above pass.
- JAX 0.11 Jaxpr.jaxpr is self-referential; the test-only program walker now
  handles equations before following that property. Initial doc builds exposed
  short title underlines and sandbox inventory DNS restrictions; corrected
  headings and a native inventory-enabled rebuild pass.

## 2026-09-15 — Uninterrupted versus restarted integration

- [x] New full-State equivalence test uses coupled island
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
- [x] Moved all 12 island scenario controls and validation into island.py.
  scenario_overrides configures initialization; step(cooling=...) supplies custom
  cooling. Generic Settings and State contain no island-only controls.
- [x] Centralized shared model/metadata/geometry/diagnostics types in _typing.py.
  Per the user's explicit correction, Settings stays with SETTINGS in
  configuration.py and PhysicalConstants stays with PHYSICALCONSTANTS in
  physical_constants.py. IslandSettings remains local to the setup.
  Removed state.py, _bulk_types.py, _solver_types.py and _thermodynamic_types.py;
  migrated consumers and documentation.
- Rejected approach: putting Settings/PhysicalConstants in _typing.py exceeded
  the requested ownership boundary and required unnecessary lazy defaults.
  Restored eager registry defaults and removed the deferred machinery/tests.
  Added expected-red tests explicitly enforcing the two local class exceptions.
- The first 123 focused cases, static checks and docs passed before this
  correction. Interrupted the obsolete full CPU run at 359 passing tests;
  those results are not final-source validation. No numerical tolerances changed.
- Final consumer audit found saltOcn_ref was used only by the island setup;
  moved its unchanged 34.7 default to ISLAND_SETTINGS and tested scenario
  overrides. Interrupted the second obsolete CPU run at 326 passing tests.
  An AST audit of every registry key found no other island-only consumers.
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
  dtypes; growth/advection/EVP scratch arrays and island masks inherit State
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
  `eps2` remain numerical or execution configuration. Island forcing and
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
  cloud tables, numerical controls and island experiment defaults are
  centralized. Exact derived values recompute on immutable replacement;
  independently rounded legacy defaults are preserved and checked against a
  historical 131-value JSON oracle from c39447e.
- [x] State is a frozen, registered 70-array JAX PyTree allocated from VARIABLES.
  Removed 14 unused/output-only fields. Five coupling outputs are returned in
  separate Diagnostics with output metadata. No configuration or mesh is in AD
  State. h5netcdf round-trip tests verify usable dimensions and attributes.
- [x] Migrated all maintained physics, structural protocols, geometry adapter,
  island integration, tests and benchmark callers to separate constants.
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
- Failed migration approaches: a 59-test fast sample missed old island
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
  integration with island ocean masks, remaining boundary/gradient gaps,
  multi-device/GPU verification. Generated version metadata is still included
  in the reported whole-package denominator.

## 2026-09-08 — geometry and standalone coupled integration

- [x] Initialization tests first failed importing Veros. Native JAX arrays in
  the host initialization routine remove that unnecessary dependency.
- [x] Independent corner-area tests then failed 2/3: rAz used two neighbors
  divided by four. Restored the four-neighbor average; uniform grid area and
  nonuniform explicit-index geometry checks now pass.
- [x] Added tests before the new `veris.setup.island` example. An island
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
  the island example supplies the requested standalone coupled path without
  claiming equivalence to a full ocean simulation.

- [x] Full suite **431/431 passed in 52.43 s**; whole-package coverage
  **886/1567 statements (56.54%)**. Initialization 3/3 and island integration
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
  single-GPU XPlane/Perfetto traces for a 64x64 island coupled step with
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
  added to island.step with dynamic cooling. Three CPU tests pass: all-field
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

- Active goal: adapt dynamics/growth experiments and parallel runner from the
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

### 2026-09-17 — Island setup and separate model setup documentation

- [x] Renamed the setup module to `veris.setups.island`, its frozen scenario
  class to `IslandSettings`, registry to `ISLAND_SETTINGS`, and prefixed
  scenario controls to `island*`. Updated callers, tests, benchmarks and docs.
  This is a direct rename; callers must use the new module and control names.
- [x] Replaced the combined experiment page with Dynamics only, Growth only,
  and Parallel dynamics pages under Model setups. Each page has its own
  generated scenario table, usage instructions and relevant validation notes.
- [x] Independent review found no correctness blockers. Retained reference
  provenance in the gallery and tailored growth documentation after review.
- [x] Sphinx warnings-as-errors build passed after allowing the external Python
  inventory fetch; initial restricted-network build failed only on that fetch.
- [x] Full CPU suite: 908 passed, 2 skipped, and one sandbox socket-permission
  failure. The affected two-process reduction/AD test passed with socket access
  (1/1), giving 909 passing tests overall. Maintained coverage: 2654/2732
  (97.14%); whole-package coverage: 2654/3085 (86.03%). No code fix was needed.
- [x] CI-scope Ruff, formatting, annotations, ty and diff checks pass. Rendered
  pages verified: independent tables contain 15 dynamics, 13 growth and 15
  parallel dynamics controls. Logs/build retained under `test_logs/island-*`.

### 2026-10-08 — Python API reference

- [x] Added a Python API navigation section with source-derived
  signatures and docstrings for model data, initialization/setup steps,
  integration, physics kernels, halo/parallel helpers and public I/O. It
  contains only Veris objects; no unit tests or test helpers are documented.
- Registry descriptions/defaults remain generated on the existing reference
  pages, linked from the API. Moved the canonical rollout/island/ocean function
  entries to the API to avoid duplicate Sphinx targets.
- First Sphinx build exposed duplicate setup targets, two short underlines,
  and RST formatting errors in three previously unrendered atmospheric-flux
  docstrings. Corrected those docstrings without changing numerical behavior.
- [x] Clean Sphinx `-E -W` build passes. Rendered audit verifies all 101
  explicit API targets and every callable parameter, 791 local links/fragments,
  unique HTML IDs, complete physics/public exports, compiled setup wrappers,
  scenario registries, navigation and absence of test API targets.
- [x] Independent documentation review found no remaining issues. Maintained
  Ruff, formatting, annotation and ty checks and `uv pip check` pass using
  the existing `.venv-latest` with `uv/latest` loaded through the module system.
  AST comparison verifies source changes are docstrings only.
- [x] Documented 8x8 island example completes three steps with finite float64
  State. Full CPU correctness suite passes: 912 passed, 2 skipped; maintained
  coverage clears the 80% gate. No new tests or numerical changes were added.
- Work remains on `jax-only` per user request; push to remote `jax-only` after
  successful verification. Artifacts: `test_logs/python-api/`,
  `test_logs/python-api-sphinx.log` and `test_logs/python-api-tests.log`.

### 2026-10-09 — Model and output precision documentation

- [x] Added a short Floating-point precision section and four-row table to
  the input/output guide, covering model dtype, instantaneous/final snapshots,
  float64 time averages and float64 time coordinates/bounds. Explained averaging
  accuracy, separation from model evolution and storage cost for float32 runs.
- Prior source audit and focused CPU precision/output checks passed 8/8 using
  `.venv-latest` and module-provided `uv`; no numerical behavior changed.
- [x] Independent documentation review found no issues. Sphinx `-E -W` build
  and rendered four-row table verification pass. Initial build failed only on
  the sandbox-blocked Python inventory fetch; rebuild with network access passed.
- [x] Fast CPU correctness sample: 92/92 passed using `.venv-latest` and
  module-provided `uv`. Diff checks pass. Build and test artifacts:
  `test_logs/precision-docs/`, `precision-docs-sphinx.log`,
  `precision-docs-tests.log` and `precision-docs-tests.xml` under `test_logs/`.
- [x] Pre-commit full CPU correctness suite: 912 passed, 2 GPU-only skips,
  no failures/errors, with socket access for multiprocess checks. Maintained
  coverage: 2654/2732 = 97.14%; the 80% gate passes. Full validation artifacts:
  `test_logs/precision-docs-full-tests.{log,xml}` and
  `test_logs/precision-docs-full-coverage.json`.

### 2026-10-12 — Notebook setup initialization and shared physics

- [x] @root completed notebook initialization/shared-physics integration and
  coordinated verification on `jax-only`. @reference_cleanup completed the
  source-derived initialization/dynamics captures; @documentation_cleanup
  completed setup/settings/lifecycle documentation and independent review.
- User confirmed the current shared Growth physics is the requested behavior.
  Both setups retain notebook initial inputs and numerical controls. Growth
  advances 150 daily steps with recursive Qnet/Qsw; dynamics retains its fixed
  wind snapshot and full momentum/transport sequence.
- Reference `jax_halo_exchange` remains
  `2571998f8feb6a5943b214361c122ca78b44c3a8`, verified through GitHub. All 130
  shared settings/constants agree with the original source.
- Keep all 70 notebook initial fields, including zero unused growth metrics
  and zero unused dynamics theta/ATemp, without altering general allocation
  defaults. Daily timesteps/five categories are explicit growth preset defaults;
  caller overrides continue to take precedence.
- [x] Source capture now retains 356 initialization/dynamics arrays and no
  growth trajectory. Regeneration verifies all retained array bytes against the
  preceding archive, 18 original source hashes and generator/archive hashes.
- Preserve corrected all-field lifecycle oracles, adapting only the known
  inactive initialization literals. Shared-default/configuration and corrected
  trajectory regressions failed 3/8 before the fix; focused checks now pass
  104/104. Failure diagnostics retain aggregate errors without dumping arrays.
- [x] Growth, Configuration and complete-restart implementations match
  pre-change commit `666775c` byte for byte. Production differences from that
  revision are confined to the two setup initializers and preset defaults.
- [x] Independent code/test review finds no remaining issue. Maintained Ruff,
  formatting, annotation and ty checks pass; clean Sphinx `-E -W` build passes.
  Evidence is under `test_logs/shared-physics/`.
- [x] Default CPU and GPU growth CLI runs both complete 150 daily steps and
  write 151 finite records with first zero ice at day 140. CPU matches the
  preceding shared-physics trajectory; CPU/GPU maximum ice difference is
  2.22e-16 m and maximum scaled difference across eight outputs is 4.50e-15.
  Saved `growth-decay.png` and `runtime-validation.json` alongside the logs.
- [x] Full CPU correctness suite passes: 1,342 passed, two GPU-only skips,
  zero failures/errors. Maintained coverage is 4,061/4,294 = 94.57%; the 80%
  gate passes. Evidence: `full.log`, `full.xml`, `coverage.json` and
  `coverage-gate.log`.
- [x] GPU correctness passes 50/50 with zero skips/errors: notebook startup
  settings/fields, 100-step dynamics, all-field shared-physics lifecycle
  references, thermodynamic conservation, longwave AD/finite differences,
  parallel CLI and two-device scheduled output. Evidence: `gpu.log`, `gpu.xml`.
