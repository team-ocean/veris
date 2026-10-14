# Model lifecycle and verification design

The public interface separates experiment definitions, numerical calculations
and the resources used to manage a run. Definitions are immutable: they cannot
change after creation. Pure numerical functions return results without changing
their inputs or external resources. The host is the Python process that manages
the run; computing devices perform the array calculations. `Experiment` groups
grid/geometry, initial conditions, forcing, numerical settings, physical
constants, execution and output requirements. Each group keeps its own role.
Geometry describes cell sizes and layout. Forcing means supplied inputs, such as
wind and heat, that drive the model. `Grid` always counts global physical cells,
the cells across the whole model area. Configuration keeps the sizes of local
partitions, the parts of the grid assigned to devices.
Definitions fill in defaults from the existing SETTINGS, PHYSICALCONSTANTS,
VARIABLES and setup scenario registries. A registry is a shared list of known
controls or fields. Python and versioned JSON use the same validation and codec
(code that converts definitions to and from their stored form).
Authored settings/constants/scenario/forcing contain only supplied values and
cannot change. Separate resolved views hold the numerical, scenario and forcing
values after defaults are filled in. Grouped Python overlays and file overrides
combine nested mappings by key and replace whole values where required. A mapping
stores values under named keys. Resolving again starts from the original authored
values. Scenario changes therefore refresh inherited defaults. Increasing dtype
precision, the detail the number format can store, does not keep defaults rounded at
lower precision. Precedence is shared defaults < preset < experiment configuration < explicit
Python/CLI overrides; rounded inputs deliberately supplied by the author stay rounded.

`initialize_experiment` checks the definition before selecting devices. It reuses
preset field factories (functions that build starting fields) or the ocean
adapter, which accepts supplied ocean data. It applies physical inputs and
prepares geometry, calculated quantities and halos with rules for each field.
Halos are extra cells around a grid block that hold values needed for neighboring
calculations. Setup modules declare functions that build preset fields for a
region. Callers use `Experiment.builtin` with grouped grid, settings, scenario
and constants. Runnable definitions share checks for the coupled physics clocks
and the host clock. `veris.grid.regions` constructs only packed partitions that
the current process can access. It caches all fields returned by each regional
callback, a function called to supply a region's data. Global indices keep the
island and spatially varying forcing in the correct places. Internal boundaries
in the north/south direction exchange values; they are not extra closed walls.
Each partition has two halos per side. Global geometry/input arrays belong to
the host definition. Each rank (distributed process) creates its local numerical
State; it never creates the whole global State and then slices it.

`advance` in `veris.execution.advance` performs exactly one physical timestep,
one advance of the model clock. It applies supplied forcing and calls unchanged
dynamics_transport. The order remains momentum, ocean stress, transport, cleanup
and ridging, followed by Growth for coupled physics. Momentum calculations
determine ice motion. Ocean stress is the force per area exchanged with the
ocean. Transport moves ice and snow; cleanup removes invalid or too-small values;
ridging accounts for ice piling up. Growth calculates freezing and melting.
Thermodynamics-only (freezing and melting) and dynamics-only (ice motion and transport) modes
follow the same rule. Coupling Diagnostics remain separate from State, which
contains only arrays. `step` applies a supplied transition repeatedly in a JAX
rollout, a sequence of model steps. Preset factories build the starting fields;
the shared `advance` transition controls their timestep sequence.

`Simulation` provides a small host interface around initialization, advance and
the existing rollout and timing code. It tracks total completed steps and model
time, manages optional scheduled output and reliably closes resources. Its pure
`(State, Diagnostics)` carry, the values passed from one step to the next, stays
inside the rollout. File handles, clocks, device mesh resources and output sums
never enter State. A mesh arranges the devices that share the grid. Constant
forcing is prepared once. Slices from the forcing series use absolute step
positions and remain inputs that can change at runtime. Spatial series split
only the physical x/y cells across partitions; the time axis stays whole. The
host updates State and time only after device synchronization confirms that
calculations have finished successfully. A failed check before execution leaves
the interface usable. An execution failure closes resources it owns. Supplied
meshes and distributed services are borrowed, so their caller retains ownership.

Scheduled output reuses io's calendars, event planner, device float64 reductions
(running sums over samples), sampling and collection across processes. Instant samples
use values after a step. Means use values before a step for half-open intervals,
which include the start time and exclude the end time. Successive host runs keep
unfinished sums/counts and the original output frequency. Optional pure
observations follow the same model trajectory, the sequence of evolving states.
Default output keeps no timestep history. Selected input checks every selected
physical value first. It then updates a candidate State and its calculated fields
and halos before accepting the result. Snapshots remain output for viewing
selected fields.

Version 1 complete restart has a separate contract for a serial CPU or a single
GPU. It restores every native State field, including the exact halos after
evolution, resolved definition/configuration/constants, complete forcing and the
absolute integer clock. It checks the file's schema (required structure) and data
before asking JAX to list devices. Typed definition arrays use the shared
experiment codec with a private array callback. This avoids conflicting storage
formats and large copies of JSON lists. The writer closes a temporary netCDF
file, then publishes it in one atomic operation without overwriting a file.
Missing or incompatible fields never use allocation defaults. Diagnostics reset
until the next step. New output keeps the saved origin/calendar/frequency, skips
the checkpoint instant and discards unfinished mean windows that are old or span
the checkpoint. Distributed restart is unsupported.

Pure advance/step support validated JIT, grad, JVP and VJP for initial State and
JAX forcing arrays. JIT compiles calculations for execution on a device. AD,
automatic differentiation, calculates how results change when inputs change.
grad calculates gradients of scalar results; JVP follows a chosen input change
forward; VJP carries result sensitivities back to inputs. Definition building,
host management, files and static settings/constants stay outside AD. Numerical
equations, important arithmetic order, masks and derivative rules at branches
remain unchanged. Masks identify where a calculation applies. Branch rules
define derivatives where the model chooses between different calculations.
Backend and distributed setup happen explicitly before devices are initialized;
imports change no global JAX settings. Local CPU sharding (splitting arrays
across devices) checks the communication, halo and output contracts. GPU and
actual multi-host checks require the corresponding hardware. Explicit launches
select devices without restricting the caller's available backends, the CPU or
GPU systems JAX can use. Hybrid multi-slice meshes use explicit x/y axes. They
match local meshes and the rules for redistributing spatial forcing while
preserving derivatives. Known owned meshes provide at least one device per
distributed process; CPU launches use exactly one per rank. These logical checks
run before launch policy and startup. Borrowed services may supply multiple
devices per rank. Preset precision applies to numerical defaults before the
final scenario is normalized. Defaults supplied by functions must remain stable
under that precision. Authored settings, constants and scenario containers are
owned immutable inputs, including their nested contents.

Independent reference trajectories are saved unchanged from jax-only commit
fff4cd4e08613e8ec2359e316c94435e81682400, before the centralized timestep implementation.
Seven cases keep all 70 State fields at steps 0, 1 and 3 and all five
Diagnostics at steps 1 and 3. They cover both boundaries and one float32 coupled
trajectory. A separate archive of all fields records unequal dynamics and
thermodynamic durations. It checks how the underlying numerical kernels,
the functions that do the calculations, are combined. Checksums and a
generator that reproduces the data record the source, configuration and forcing.
These saved forward results complement independent equation, stencil,
conservation and finite-difference AD tests. A stencil specifies which nearby
grid cells enter a calculation. Conservation checks track physical budgets;
finite differences estimate derivatives from small input changes. Comparing
two callers of advance alone cannot establish scientific equivalence.

## Numerical test harness

The `jax-only` source is the initial numerical reference. Tests belong in
`tests/`, follow the physics module layout and execute real JAX kernels. Run
checks of validation, metadata (descriptions of fields), shapes, masks and
indexing once when they do not depend on precision. Use float64 for precise
equation and gradient references, and float32 for representative runs and
stability checks. These formats store numbers with 64 or 32 bits. Use both only
for dtype propagation, conversion, sensitive thresholds and a small integration
sample. Dtype propagation means keeping the intended number format through a
calculation. Use frozen dataclasses, Python data containers that cannot change,
for separate static Configuration and PhysicalConstants and for the minimal
array-only State PyTree. A PyTree groups arrays in a structure that JAX can
process. Initialize defaults from SETTINGS, PHYSICALCONSTANTS and VARIABLES.
Their metadata also generates reference documentation. Output-only coupling
diagnostics stay outside State. Compare against independent scalar equations,
stencils written with explicit cell indices, conservation laws and finite differences.

Use small rectangular grids to reveal axis errors. Test masks, zero forcing,
thresholds, hemispheres and inputs that vary across cells. Never replace JAX or
physics with fake modules to obtain coverage. Test coverage measures how much
code the tests run. Treat distributed initialization as a separate boundary
between systems. Test CPU by default; support GPU through JAX configuration.

CPU correctness defaults to module-grouped xdist execution with up to three
workers, allowing four permitted CPUs per worker when available. Smaller or
unsupported allocations keep one worker. On supported hosts, disjoint affinity is set
before backend initialization; each worker retains its private cold cache.
Twelve permitted CPUs select three workers; explicit worker counts remain
caller policy. GPU/unspecified platforms keep one worker without early device
discovery. Use `-n 0` for focused development checks. Full runtime is measured
externally with the CPU/memory allocation recorded alongside correctness.

Pytest `--fast` selects a stable approximately 10% sample when it collects tests.
The sample depends on `VERIS_TEST_SEED` and each test's node ID. Run full
correctness and coverage checks before any commit. Report whole-package coverage,
including generated metadata. With user approval, enforce the 80% CI target on
maintained code by omitting only generated `veris/_version.py` from that check.
CI means continuous integration, the repository's automated checks. The old
geographic setup is removed from scope and source at the user’s explicit request.
CPU CI enforces these checks, documentation builds that reject warnings and
public examples run against a clean installation.

Implementation sequence: prepare reusable test inputs and basic numerical tests;
add halo and transport checks; test dynamic and thermodynamic equations and
gradients where the equations vary smoothly; review test quality, measure
coverage and close gaps. Add tests that reproduce known broken branches before
fixing the model code.

Each type has one home: Configuration and SETTINGS live in `veris.model.configuration`;
PhysicalConstants and PHYSICALCONSTANTS in `veris.model.physical_constants`;
shared Parameter metadata, State and kernel input/output rules in `veris.model.types`;
Variable in `veris.model.variables`; Diagnostics in `veris.model.diagnostics`.
Precision metadata is defined directly in SETTINGS["dtype"]. Shared ArrayInput
and MaskInput contracts live in `veris.model.arrays`. OceanGeometry belongs to
`veris.grid.geometry`.
Island-only configuration remains in `veris.setups.island`.
Configuration retains the unused `explicitDrag` setting because version-one
complete restart files store its full settings schema. Keeping this field
does not enable an alternative drag calculation.

The model prepares external ocean geometry from scratch through
`veris.grid.ocean.initialize_from_ocean` and the shared
`veris.execution.initialization.initialize`.

The main dynamics sequence lives in
`veris.physics.dynamics.driver.dynamics_transport`. It owns
momentum, ocean stresses before transport, advection (movement with the ice),
cleanup and ridging. The pure `veris.execution.advance` transition applies
supplied forcing, thermodynamics and the final halo rules for each field.
Simulation manages host execution contexts, time and I/O (input and output).
Tests check the expected stage sequence separately from host management and
independent numerical kernels.

## Scientific package hierarchy

The implementations are grouped by their scientific role and function:

- `veris.model`: immutable definitions, numerical settings, physical constants,
  State and array input/output rules, diagnostics, registries and the experiment codec.
- `veris.grid`: geometry schema and ocean adapter; interpolation between
  staggered grid positions, where quantities occupy different locations in a
  cell; two-cell halos; reductions; and regional inputs the process can access.
- `veris.physics.dynamics`: mass, rheology (how ice deforms under force), ocean
  stress and the ordered dynamics stage; `veris.physics.transport`: advection,
  cleanup and ridging; `veris.physics.thermodynamics`: growth and atmospheric
  bulk heat fluxes, estimates of heat exchange with the air per area and time.
- `veris.numerics`: basic differentiation operations that handle branches
  safely, momentum solver selection, EVP (elastic-viscous-plastic ice motion),
  free drift (motion without internal ice stress) and repeated calculations
  to find surface temperature.
- `veris.execution`: array allocation, preparation of runnable experiments,
  the pure timestep and rollout, timed output rollouts and the host Simulation lifecycle.
- `veris.io`: calendars, scheduling, storage, collection, run I/O and
  complete restart with typed definition arrays.
- `veris.setups`: maintained island, dynamics, growth and parallel entry points
  and preset controls.

Package initializers explain these roles without importing execution into lower
layers. Geometry depends only on shared array aliases, names for array types.
I/O uses `veris.grid.halos.remove_halos` for packed physical fields rather than
importing the parallel setup driver. The experiment codec/preset imports happen
only when needed. This keeps the convenient host interfaces and avoids circular
imports, where modules need each other before either has finished loading.

Each implementation has one home. The root `veris` interface exposes the
experiment and execution interfaces; `veris.io` exposes host output and record
interfaces. Setup modules prepare presets and supply command-line examples.
Generated version metadata stays at the package root. The complete tree and
standard import examples are maintained in
[package-hierarchy.rst](docs/reference/package-hierarchy.rst). Scientific capture
records keep the source paths from the archived revision used to produce them.

The host-only `veris.cli` Click command group is available through the installed
`veris` command and `python -m veris.cli`. Click builds command line interfaces.
Discovery/copy never imports template code or configures JAX. Builtin/external
folders contain editable `setup.py` factories returning Experiment. Run loads
those factories or the existing versioned JSON codec, applies explicit
execution/precision overrides and manages Simulation/mesh/distributed resources.
When options are omitted, the definition's values apply. Float overrides update
numerical and physical precision together. Snapshot/history use existing
physical-field I/O and protect existing files from overwrite. Numerical kernels
and preset initializers are not duplicated. Run and inspect also accept
builtin/external names from the same registry as show-setups and copy-setup. A
shared host resolver checks existing files first, rejects missing explicit paths
and looks up names without executing factories.

Tests follow the domain package layout. Standalone process probes (checks run in
separate processes) and immutable reference data keep their shared locations.
Full correctness and coverage checks use one coordinated pytest session. Elapsed
time is measured outside that session. Test selection, scientific tolerances,
unchanged reference bytes and required numerical/gradient assertions remain part
of the correctness contract. Tolerances set the allowed numerical differences.

Experiment delivery contract
----------------------------

All CLI/module run entry points use `veris.run_experiment`. They apply explicit
grouped overrides to authored inputs before filling in defaults. Provider
factories supply regional data. Shared initialization checks inputs, calculates
derived fields and applies global boundary rules. Runtime resources keep
distributed/mesh ownership outside numerical State. Authored exports and resolved
exports with fixed values use the same codec; version-one definitions/checkpoints
remain readable. The executable custom-ocean setup blueprint and Python/CLI
probe against a clean installation check the documented interface.
