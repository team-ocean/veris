# Development log

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
  source/runtime changes. Remote revalidation pending.
