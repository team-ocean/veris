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
derived quantities and field-aware halos. `model_regions` constructs only
addressable packed partitions, caching all fields per regional callback. Global
indices preserve the island location and spatial forcing; internal meridional
boundaries are exchanges rather than extra closed walls. Each partition has two
halos per side. Global geometry/input arrays belong to the host definition;
numerical State is never globally initialized per rank and then sliced.

`advance` in model_physics performs exactly one physical timestep. It applies
supplied forcing, calls unchanged dynamics_transport in the established momentum,
ocean-stress, transport/cleanup/ridging order, then Growth for coupled physics.
Thermodynamics-only and dynamics-only modes share this contract. Coupling
Diagnostics remain separate from the array-only State. Existing public `step`
remains a supplied-transition multi-step JAX rollout; compatibility setup drivers
delegate their timestep sequence to advance.

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

Pure advance/step support validated JIT, grad, JVP and VJP with respect to initial
State and JAX forcing arrays. Definition construction, host orchestration, files
and static settings/constants stay outside AD. Numerical equations, significant
arithmetic order, masks and branch derivative conventions remain unchanged.
Backend/distributed setup is explicit before devices; imports change no global
JAX settings. Local CPU sharding establishes communication/halo/output contracts;
GPU and actual multi-host verification require the corresponding hardware.

Independent legacy trajectories are frozen from jax-only commit
fff4cd4e08613e8ec2359e316c94435e81682400 before compatibility wrappers delegate.
Seven cases retain all 70 State fields at steps 0, 1 and 3 and all five
Diagnostics at steps 1 and 3, both boundaries and one float32 coupled trajectory.
A separate all-field reference protects legacy unequal island timesteps.
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

Type ownership: Configuration and SETTINGS live in configuration.py;
PhysicalConstants and PHYSICALCONSTANTS in physical_constants.py;
shared Parameter metadata in _typing.py; Variable in
variables.py; Diagnostics in diagnostics.py. Precision metadata is defined
directly in SETTINGS["dtype"]. State, OceanGeometry and shared kernel contracts
live in _typing.py. Island-only configuration remains in setups/island.py.

External ocean geometry is initialized from scratch through
setups/ocean.py:initialize_from_ocean and the shared initialization.initialize.

Core dynamics orchestration lives in dynamics.py:dynamics_transport. It owns
momentum, pre-transport ocean stresses, advection, cleanup and ridging. The pure
model_physics transition applies prescribed forcing, thermodynamics and the
final field-specific halo policy. Compatibility setups delegate that shared
sequence; Simulation owns host execution contexts, time and I/O. Tests separate
the stage sequence oracle from host wrapping and independent numerical kernels.
