# Test harness design

The `jax-only` source is the initial numerical reference. Tests belong in
`tests/`, mirror physics modules, and execute real JAX kernels with float64
arrays. Use frozen dataclasses for separate static Settings and PhysicalConstants and
for the minimal array-only State PyTree. Initialize defaults from SETTINGS,
PHYSICALCONSTANTS and VARIABLES; metadata also generates reference documentation.
Output-only coupling diagnostics stay outside State. Compare against independent scalar equations,
explicit index-based stencils, conservation laws, and finite differences.

Use small rectangular grids to expose axis errors; exercise masks, zero forcing,
thresholds, hemispheres, and nonuniform inputs. Never replace JAX or physics with
fake modules to obtain coverage. Distributed initialization is an explicit
integration boundary. Test CPU by default; support GPU through JAX configuration.

Pytest `--fast` selects a stable approximately 10% sample at collection, keyed by
`VERIS_TEST_SEED` and node ID. Full correctness and coverage checks precede any
commit. Report whole-package coverage including generated metadata. Per user
approval, enforce the 80% CI target on maintained code by omitting only generated
`veris/_version.py` from that gate. The old geographic setup is removed
from scope and source per the user’s explicit instruction. Add CI once the harness is executable.

Implementation sequence: establish fixtures and elementary numerical tests;
add halo and transport checks; test dynamic and thermodynamic equations and
smooth-region gradients; review test quality, measure coverage, and close gaps.
Known broken branches need reproducing tests before production fixes.
