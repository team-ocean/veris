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
