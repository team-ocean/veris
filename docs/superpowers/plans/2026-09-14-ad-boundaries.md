# AD boundary repairs

Goal: repair zero-strain AD and related avoidable NaN/Inf sensitivities across
Veris dynamics, transport and thermodynamics, including original setup states.

Preserve reference forward values and smooth-region derivatives wherever
possible. Guard singular operations before evaluation rather than masking NaN
outputs or gradients afterward. For true norm kinks, explicitly select zero
linearization at the origin and test symmetric differences; do not claim that
this creates a classical derivative. Do not change thresholds/tolerances to
hide regressions. Any necessary forward-equation change requires independent
physical justification and explicit oracle updates.

- [x] Reproduce zero-strain, zero-speed, masked and zero-forcing AD failures.
- [x] Repair dynamics norm/floor/relaxation paths and free-drift boundary cases.
- [x] Audit and repair transport, averaging, growth and atmospheric branch hazards.
- [x] Validate JVP/VJP, finite differences, actual initial case trajectories,
      CPU/GPU and sharded execution; preserve numerical reference oracles.
- [x] Independent review, documentation/CHANGELOG, full suite and checks, commits.

Root schedules the only pytest instance. ad_thermo_audit performs read-only
thermodynamic audit. Current worktree starts clean except local runtime logs.
