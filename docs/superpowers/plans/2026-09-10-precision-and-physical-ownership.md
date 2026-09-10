# Initialization precision and physical ownership

Goal: select floating-point precision once at initialization and propagate it to all
state arrays, static coefficients, derived coefficients, tables and temporary arrays.
Physical closure coefficients belong to PhysicalConstants; execution choices,
numerical solver controls, grid sizes and experiment inputs belong to Settings.

Implementation follows the user's authorized continuation on jax-only.

- [x] Reproduce missing physical ownership and dtype selection with tests.
- [x] Move physical thresholds and their validation, kernel consumers and tests.
- [x] Define common immutable precision metadata in _metadata.py. Inherit the
  keyword-only dtype field in both static dataclasses; keep it outside State.
  Normalize floating inputs and derived values at construction, preserving
  integer/Boolean controls and hashable static arguments.
- [x] Allocate state and work arrays using selected/inherited precision.
- [x] Verify full coupled steps and AD in float32 and float64, x64-disabled
  initialization, invalid dtypes, and representability failures.
- [x] Update metadata-driven netCDF examples and precision documentation.
- [x] Run maintained lint/type/docs checks, full CPU/GPU correctness suites,
  independent review and record results in CHANGELOG.md before committing.

Precision policy is shared metadata rather than a physical constant. Its default
is defined once in PRECISION. Variable metadata supplies shapes, units and defaults;
netCDF storage dtype comes from the actual allocated array. Selecting float64
requires JAX x64 support already enabled; initialization never changes global JAX
configuration. Immutable replacement preserves the object's selected precision.
