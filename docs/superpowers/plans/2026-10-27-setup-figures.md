# Setup result figures implementation plan

## Revised execution plan

- [x] Add failing tests for same-area 80 by 120 island refinement, regional
  center/face masks, invalid half-width and growth plotting through first zero.
- [x] Add `islandHalfWidthCells` and the `refined` profile to
  `veris/setups/island.py`, preserving original defaults and references.
- [x] Run dynamics 300 steps and growth 200 steps; run refined island 300
  steps and check trajectory CFL, masks, finite fields and solver/timestep
  sensitivity at equal duration using existing Simulation observers.
- [x] Add `docs/examples/plot_setup_results.py`; generate reviewed SVG/PDF/
  600 dpi PNG figures and numerical provenance. Remove superseded figures.
- [x] Update all three setup pages with exact commands, settings, correct
  growth window, figure/download links and island area/timestep rationale.
- [x] Run new/focused tests and one complete fast sample, maintained static
  checks and strict Sphinx; inspect scientific figures and PDF renders.
- [x] Complete independent scientific/code/documentation review; update
  CHANGELOG.md and leave a verified uncommitted result.

The explicit user clarification requires 256 square km of island land before
and after refinement. The `refined` profile uses 800 m cells and half-width ten,
preserving both basin and island geometry. Provisional numerics: dt60 s,
adaptive EVP400, compared with EVP800 and dt30 s at equal physical duration.

Completed: maximum directional CFL sum 0.01174; concentration sensitivity max
0.0017032 for EVP800 and 0.0005668 for dt30 s at equal duration. Growth is
plotted through day154 (155 of 201 stored samples). TDD red5, focused62+2pass,
fast176pass, maintained static checks and strict Sphinx pass. All nine exports
and final PDF renders pass visual/data review. Reviewer-found introductory
output-path collisions are corrected. Final changes remain uncommitted.

## Original execution record

> **For agentic workers:** Execute this documentation-only plan with
> superpowers:executing-plans and an independent scientific/documentation review.

**Goal:** Add real 100-step dynamics/island concentration maps and a 150-day
growth thickness time series to their corresponding setup pages.

**Architecture:** Existing CLI runners produce physical-cell netCDF snapshots
and a growth history. Matplotlib renders checked output; Sphinx embeds static
figures without executing simulations during documentation builds.

**Tech stack:** Veris Experiment/Simulation, JAX CPU float64, h5netcdf,
Matplotlib, reStructuredText, Sphinx.

**Spec:** `docs/superpowers/specs/2026-10-27-setup-figures-design.md`.

## Constraints and review focus

Preserve preset grids, timestep lengths, solver iteration counts, physics and
user edits. Verify `(x, y)` orientation, masked land including dynamics wall
cells, physical coordinate spacing, 151 growth samples including the initial
state, and grid-cell mean thickness. Do not add tests that mirror prose edits.
Use existing tests and direct numerical/output verification.

## Tasks

- [x] Read DESIGN.md, CHANGELOG.md, AGENTS.md, relevant setup/CLI/I/O code and
  documentation; inspect the clean checkout and GitHub repository identity.
- [x] Run dynamics with `--steps 100`, growth with `--steps 150` and daily
  instantaneous history, and island with `--steps 100`; retain preset defaults.
- [x] Validate output shapes, times, masks, finite values and physical bounds;
  save compact provenance in `docs/validation/2026-10-27-setup-figures.json`.
- [x] Render and visually inspect three PNGs under `docs/figures/setups/`.
- [x] Add figures, captions and exact run commands to
  `docs/reference/setups/dynamics.rst`, `growth.rst` and `island.rst`.
- [x] Run strict Sphinx and `pytest tests/ -q --fast` with temporary products
  under `/private/tmp`; inspect the three built pages and image references.
- [x] Review the final diff independently, resolve findings and record results
  in CHANGELOG.md before delivering the local documentation changes.

## Execution record

Growth's measured final thickness is 0.16912611295764002 m. A run without
scheduled output matches exactly, so replace the stale day-140 disappearance
claim with this current result. Do not change thermodynamic equations.

Strict Sphinx passes after allowing its Python inventory request. The fast
sample has 175 selected tests: 173 pass initially and both sandbox-blocked
localhost tests pass on exact rerun with access. Maintained CI static checks
pass; root-wide checks retain pre-existing vendor/generated findings.

Review the PNGs visually and built HTML structurally: no browser is available
for page screenshots. Keep all edits in the requested working directory and
leave them uncommitted; the user requested figures and documentation, without
remote publication or source changes.

Final independent review finds no Critical, Important or Minor content issues.
CHANGELOG.md records numerical results, successful checks, environmental
reruns, existing root-wide static findings and the verification limits.
