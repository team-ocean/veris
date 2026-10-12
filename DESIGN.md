# Model lifecycle and verification design

The public interface separates immutable experiment definitions, pure numerical
execution and host lifecycle resources. `Experiment` groups grid/geometry,
initial conditions, forcing, numerical settings, physical constants, execution
and output requirements without merging their responsibilities. `Grid` always
counts global physical cells; Configuration retains local partition extents.
Definitions resolve existing SETTINGS, PHYSICALCONSTANTS, VARIABLES and setup
scenario registries. Python and versioned JSON share the same validators/codec.

`initialize_experiment` validates before selecting devices, reuses preset field
factories or the ocean adapter, applies physical inputs and prepares geometry,
derived quantities and field-aware halos. `veris.grid.regions` constructs only
addressable packed partitions, caching all fields per regional callback. Global
indices preserve the island location and spatial forcing; internal meridional
boundaries are exchanges rather than extra closed walls. Each partition has two
halos per side. Global geometry/input arrays belong to the host definition;
numerical State is never globally initialized per rank and then sliced.

`advance` in `veris.execution.advance` performs exactly one physical timestep.
It applies supplied forcing, calls unchanged dynamics_transport in the established momentum,
ocean-stress, transport/cleanup/ridging order, then Growth for coupled physics.
Thermodynamics-only and dynamics-only modes share this contract. Coupling
Diagnostics remain separate from the array-only State. `step` performs a
supplied-transition multi-step JAX rollout. Preset factories construct initial
fields; the shared `advance` transition owns their timestep sequence.

`Simulation` is a thin host facade over initialization, advance and the existing
rollout/timing machinery. It owns absolute completed-step/time counters, optional
scheduled output and reliable resource closure. Its `(State, Diagnostics)` pure
carry stays inside the rollout; file handles, clocks, mesh resources and output
accumulators never enter State. Constant forcing is prepared once and absolute
series slices remain dynamic inputs. Spatial series partition only physical x/y,
with an unpartitioned time axis. Successful device synchronization precedes host
State/time commit. Invalid preflight leaves the facade usable; execution failure
closes owned resources. Supplied meshes and distributed services are borrowed.

Scheduled output reuses io's calendars, event planner, device float64 reductions,
sampling and distributed collection. Instant samples are post-step; half-open
means sample pre-step. Successive host runs retain unfinished sums/counts and
original cadence. Pure optional observations share the same trajectory; default
output retains no timestep history. Selected input validates the entire physical
selection, then updates a candidate and its derived fields/halos before commit.
Snapshots remain selected-field visualization output.

Version 1 complete restart is a distinct serial CPU/single-GPU contract. It
restores every native State field including exact evolved halos, resolved
definition/configuration/constants, complete forcing and absolute integer clock.
Cold schema/data validation precedes JAX device enumeration. Typed definition
arrays use the authoritative experiment codec with a private array callback,
avoiding divergent serialization and large JSON list copies. A closed temporary
netCDF file is published atomically without overwrite. Missing/incompatible fields
never fall back to allocation defaults. Diagnostics reset until the next step.
New output keeps saved origin/calendar/cadence, skips the checkpoint instant and
discards old/crossing unfinished mean windows. Distributed restart is unsupported.
The exact version-1 configuration inventory preceding useLegacyGrowthAreaWeights
is migrated with only that known control set to False, preserving the corrected
equations of older checkpoints. All other missing or incompatible metadata fails.

Pure advance/step support validated JIT, grad, JVP and VJP with respect to initial
State and JAX forcing arrays. Definition construction, host orchestration, files
and static settings/constants stay outside AD. Numerical equations, significant
arithmetic order, masks and branch derivative conventions remain unchanged.
Backend/distributed setup is explicit before devices; imports change no global
JAX settings. Local CPU sharding establishes communication/halo/output contracts;
GPU and actual multi-host verification require the corresponding hardware.

Independent reference trajectories are frozen from jax-only commit
fff4cd4e08613e8ec2359e316c94435e81682400 before the centralized timestep implementation.
Seven cases retain all 70 State fields at steps 0, 1 and 3 and all five
Diagnostics at steps 1 and 3, both boundaries and one float32 coupled trajectory.
A separate all-field archive records unequal dynamics/thermodynamic durations
and verifies canonical composition of the underlying numerical kernels.
Checksums and a reproduction
generator record source/configuration/forcing provenance. These golden primals
complement independent equation/stencil/conservation and finite-difference AD
tests; comparing two callers of advance alone is not scientific equivalence.

## Numerical test harness

The `jax-only` source is the initial numerical reference. Tests belong in
`tests/`, mirror physics modules, and execute real JAX kernels. Run
precision-independent validation, metadata, shapes, masks and indexing once. Use float64 for precise equation and gradient
references, and float32 for representative operational and stability checks.
Use both only for dtype propagation, conversion, sensitive thresholds and a
small integration sample. Use frozen dataclasses for separate static Configuration
and PhysicalConstants and for the minimal array-only State PyTree. Initialize defaults from SETTINGS,
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
from scope and source per the user’s explicit instruction. CPU CI enforces these
checks, strict documentation builds and clean installed public examples.

Implementation sequence: establish fixtures and elementary numerical tests;
add halo and transport checks; test dynamic and thermodynamic equations and
smooth-region gradients; review test quality, measure coverage, and close gaps.
Known broken branches need reproducing tests before production fixes.

Type ownership: Configuration and SETTINGS live in `veris.model.configuration`;
PhysicalConstants and PHYSICALCONSTANTS in `veris.model.physical_constants`;
shared Parameter metadata, State and kernel contracts in `veris.model.types`;
Variable in `veris.model.variables`; Diagnostics in `veris.model.diagnostics`.
Precision metadata is defined directly in SETTINGS["dtype"]. Shared ArrayInput
and MaskInput contracts live in `veris.model.arrays`. OceanGeometry belongs to
`veris.grid.geometry`.
Island-only configuration remains in `veris.setups.island`.

External ocean geometry is initialized from scratch through
`veris.grid.ocean.initialize_from_ocean` and the shared
`veris.execution.initialization.initialize`.

Core dynamics orchestration lives in
`veris.physics.dynamics.driver.dynamics_transport`. It owns
momentum, pre-transport ocean stresses, advection, cleanup and ridging. The pure
`veris.execution.advance` transition applies prescribed forcing, thermodynamics
and the final field-specific halo policy. Simulation owns host execution contexts,
time and I/O. Tests separate
the stage sequence oracle from host wrapping and independent numerical kernels.

## Scientific package hierarchy

Authoritative implementations are grouped by scientific and functional ownership:

- `veris.model`: immutable definitions, numerical settings, physical constants,
  State and array contracts, diagnostics, registries and the experiment codec.
- `veris.grid`: geometry schema and ocean adapter, staggered interpolation,
  two-cell halos, reductions and addressable regional inputs.
- `veris.physics.dynamics`: mass, rheology, ocean stress and the ordered dynamics
  stage; `veris.physics.transport`: advection, cleanup and ridging;
  `veris.physics.thermodynamics`: growth and atmospheric bulk heat fluxes.
- `veris.numerics`: branch-safe differentiation primitives, momentum dispatch,
  EVP, free drift and surface-temperature iteration.
- `veris.execution`: allocation, runnable experiment initialization, the pure
  timestep and rollout, timed output rollouts and the host Simulation lifecycle.
- `veris.io`: calendars, scheduling, storage, collection, lifecycle I/O and
  complete restart with typed definition arrays.
- `veris.setups`: maintained island, dynamics, growth and parallel entry points
  and preset controls.

Package initializers document these responsibilities without importing execution
into lower layers. Geometry depends only on shared array aliases. I/O uses
`veris.grid.halos.remove_halos` for packed physical fields rather than importing
the parallel setup driver. Deferred experiment codec/preset imports retain the
host convenience interfaces without import-time cycles.

Each implementation has one canonical owner. The root `veris` facade exports
the experiment and execution interfaces; `veris.io` exports host output and
record interfaces. Setup modules supply preset initialization and command-line
examples. Generated version metadata stays at the package root. The complete
tree and canonical import examples are maintained in
[package-hierarchy.rst](docs/reference/package-hierarchy.rst). Scientific capture
provenance records the paths of its archived source revision.

Tests mirror the domain packages while standalone process probes and immutable
reference data retain shared locations. Full correctness and coverage verification
uses one coordinated pytest session with externally measured wall time.
Test selection, scientific tolerances, frozen reference bytes and required
numerical/gradient assertions remain part of the correctness contract.
