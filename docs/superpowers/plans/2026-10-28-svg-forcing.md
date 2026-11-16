# SVG forcing and island example implementation plan

> Execute with the Superpowers TDD and independent review workflow.

**Spec:** `docs/superpowers/specs/2026-10-28-svg-forcing-design.md`.

- [x] Inspect guidance, current worktree, raw output and field staggering;
  confirm saved island ocean is stationary and obtain user's data preference.
- [x] Promote high-resolution island to the sole example/defaults and adapt
  reference tests with explicit compact controls (@sole_island_example).
- [x] Add failing numerical tests for staggered-to-centered ocean velocity
  and stationary-ocean overlays; add sparse calibrated wind vectors and
  translucent distinguishable streamlines in the plotting script (@root).
- [x] Recover omitted fixed dynamics forcing by zero-step initialization of
  its recorded definition; reuse all original integrated states/history.
- [x] Generate only three SVGs, remove old PNG/PDF outputs, remove growth
  annotation and dynamics land legend, and inspect visual results.
- [x] Update pages, commands and provenance; verify copied SVGs, responsive
  display and absence of obsolete downloads in a strict Sphinx build.
- [x] Run focused checks and one deterministic fast suite (one pytest at a
  time), maintained static checks and independent review; update changelog.

The island implementer owns pytest until its focused task completes; root
can run direct Python numerical assertions in parallel, then owns final pytest.

Final evidence: 124 control/archive checks and 298 compatibility checks pass
(two expected GPU skips); four plot tests and 176 fast-sample tests pass.
Maintained static checks and strict Sphinx pass; installed lifecycle examples
pass with the required logical CPU devices. Final independent review is clean.
The prior nine exports are now three SVG files; the six former PDF/PNG files
are removed. No model trajectory is rerun.
