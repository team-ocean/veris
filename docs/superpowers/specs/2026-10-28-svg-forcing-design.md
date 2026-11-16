# SVG forcing overlays and sole island example

User requests SVG-only figures: ocean streamlines and sparse atmospheric wind
vectors on concentration maps, no dynamics land legend, and no growth
ice-free annotation. Distinct muted green ocean lines and vermilion wind arrows
retain concentration visibility. Interpolate ocean U/V face velocities to cell
centers before plotting; wind is already cell-centered. Keep equal map aspect,
physical units, a wind scale key and a fixed zero-to-one concentration scale.

Reuse all existing final numerical output. Dynamics saved only Area/mask;
recover its fixed prescribed velocities from the snapshot's exact recorded
definition with initialization only (zero timesteps), retaining a separate
forcing snapshot and its provenance. Never use ice velocities as ocean flow.
The saved island ocean velocity is identically zero. User explicitly chooses
to keep the run and show stationary ocean, with wind vectors and no invented
current or rerun.

Promote former refined island settings to the sole `example` and ordinary
island defaults: 80x120 cells, 800 m spacing, half-width ten cells, 60 s equal
physics timesteps and adaptive EVP400. Preserve the 64x96 km basin and 256 km2
island. Remove the `refined` profile name and low-resolution public setup;
keep scientific reference checks using explicit test-only compact controls.

Export only SVG, with outlined lettering for portable web rendering; remove
the previous PNG/PDF deliverables and download links. Record exact raw output,
supplemental forcing and SVG hashes, including the historical `refined` label
in reused raw data without relabelling those files. Update documentation and
CHANGELOG.md. No commit or remote publication requested.
