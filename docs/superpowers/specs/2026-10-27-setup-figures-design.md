# Setup result figures

## Revised request: longer runs and publication/web figures

Supersedes the original run lengths below. Rerun dynamics for 300 steps and
growth for 200 daily steps. Plot growth only through its first exact zero
thickness, including that endpoint (measured day 154). Refine island from
8 by 12 to 80 by 120 cells and run 300 steps. User explicitly requires the
same island area: preserve the 64 by 96 km basin and 16 by 16 km island,
using 800 m cells and a 20 by 20 land block.

Add a named `refined` island profile rather than altering archived small-grid
defaults. A positive integer `islandHalfWidthCells` defaults to one and is ten
in this profile; both center and face masks must use it with global indices.
Reject a land block that spans either global grid extent. Initial numerical
choice: equal 60 s physics timesteps, adaptive EVP with 400 iterations. Assess
the entire trajectory and compare with 800 iterations and 30 s timesteps at
the same physical duration before accepting the figure data. These checks
measure finite-duration stability/sensitivity, not continuum convergence.

Produce restrained scientific plots with physical units, readable fonts,
equal map aspect ratios, a perceptually ordered concentration scale spanning
zero to one, masked grey land and a labelled first-zero growth endpoint.
Supply SVG for the website, PDF with embedded fonts for publication, and
600 dpi PNG exports. Retain an executable figure-generation script and tested
growth-window selection. Replace superseded 100/150-step figures and update
each setup page, downloadable assets, provenance, plan and changelog.

Validation: TDD for new mask/profile/plot-window behavior; current preset and
regional-reference tests; full-trajectory stability metrics and sensitivity
comparisons; artifact/hash checks; visual inspection including PDF renders;
strict Sphinx; one deterministic fast pytest run; maintained static checks.
Leave changes uncommitted, as no commit/push is requested.

## Original request (completed, retained for context)

Generate the three figures requested on 2026-10-27 using existing public
setup runners and unchanged preset physics. Dynamics runs 100 steps on its
1024 by 1024 standalone grid, growth runs 150 daily steps on its 2 by 2 grid,
and island runs 100 coupled steps on its 8 by 12 example grid. Use CPU float64.

## Approach

Run the maintained CLI drivers, then plot their netCDF output with Matplotlib.
This preserves the shared Experiment/Simulation lifecycle and provides actual
model results. A reduced dynamics grid would be faster but would change the
default experiment; a separate integration implementation would duplicate
existing numerical orchestration. Neither is needed.

## Figures and documentation

- Dynamics and island: final `Area`, on physical x/y coordinates in kilometres,
  transposed from native `(x, y)` storage for plotting, with land masked using
  `maskInC` and a shared concentration scale from zero to one. Remove halos
  through the existing snapshot writer, never by slicing output a second time.
- Growth: `hIceMean` at days 0 through 150 from instantaneous scheduled output;
  plot a uniform physical cell, after verifying that all four cells agree.
  Label thickness as a grid-cell mean in metres.
- Embed PNG figures with alt text, captions, run settings and reproduction
  commands on `docs/reference/setups/{dynamics,growth,island}.rst`.
- Retain compact run provenance under `docs/validation/`; keep model netCDF
  output, temporary plotting code and build products outside the checkout.

## Verification

Check actual step counts/model times, array shapes, finite physical values,
concentration bounds, masks, and the growth time axis. Inspect each rendered
figure and the built pages. Run strict Sphinx and the deterministic fast test
sample, with only one pytest process at a time. Update CHANGELOG.md. This task
does not change model code or require a commit or remote publication.
