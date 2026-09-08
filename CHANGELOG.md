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
