# Centralized Veris model lifecycle

## Intent and existing architecture

Implement the complete AGENTS.md Goal: a scientist defines, initializes, runs,
saves and resumes an experiment through public Veris interfaces. Numerical
equations, significant arithmetic order, tolerances and array-only State remain
unchanged. This is an orchestration layer, not a new physical model.

The existing allocator, geometry adapter, dynamics stage, pure scan rollout,
calendar planner, reduced output runner and netCDF storage are reusable. Current
gaps are a definition schema, runnable custom initialization, shared physics
wrapping and forcing, model-time ownership, output continuation and complete
restart. Setup CLIs duplicate application orchestration. Reference behavior is
the maintained jax-only branch, never main.

Choose a thin host `Simulation` facade over these components. Merely binding
setup callables still leaves scientists owning lifecycle correctness; a large
class containing equations and storage would duplicate proven components.

## Definition

Public immutable `Grid(nx, ny)` always counts global physical cells. Its
`local_shape(partitions)` validates divisibility and at least two cells per
partition; `storage_shape(partitions)` includes two halos on every partition
edge. C/U/V/Z fields have identical storage extents, with staggering described
by VARIABLES. Existing Configuration nx/ny retain their local convention.

`Execution(backend=None, partitions=(1, 1))` describes resource selection without
enumerating devices. None honors the caller's JAX backend selection; explicit
cpu/gpu selects a device at initialization. Distributed JAX initialization and
any process-wide backend configuration remain explicit before enumeration.
Explicit meshes stay on Simulation, outside State and the serializable definition.

`Experiment` groups Grid, separate numerical `settings` overrides, separate
`constants` overrides, physics (`coupled`, `dynamics`, `thermodynamics`), optional
OceanGeometry, initial conditions, Forcing, Execution, total steps and optional
OutputSettings. Configuration and PhysicalConstants constructors remain the
only validators/default owners for their parameters. Reject nx/ny/use_sharding
in numerical overrides, because Grid/Execution own them; constants inherit
model dtype and cannot specify a conflicting precision. Equal coupled timesteps
are required until real subcycling is implemented. Definition defaults allocate
no JAX arrays and change no JAX configuration.

Initial conditions use scalar or global physical (nx, ny) values for hIceMean,
hSnowMean, Area, TSurf, uIce, vIce, sigma1, sigma2, sigma12. Validate finite real
values, nonnegative thicknesses and concentration in [0,1]. Forcing constant
fields are scalar or global physical arrays; time series are (time,) for spatially
uniform values or (time,nx,ny). Series are step-indexed, sampled immediately
before the physical timestep, and override constant entries of the same name.
Require every series to have one shared leading length. Allowed forcing names
are maintained prescribed ocean/atmospheric State inputs, not evolved ice or
derived dynamics fields. Host definitions defensively copy mutable inputs;
pure execution accepts dynamic JAX forcing arrays separately for AD.

Use versioned JSON as the simple declarative file format (standard-library
parser, no new package dependency). Registry defaults < file values < explicit
Python overrides. Merge mappings by key, replace arrays and lists as whole
values. Reject unknown keys, malformed sections and incompatible geometry.
Resolved metadata includes numerical defaults/derived quantities, constants,
global dimensions, execution, forcing policy/data, calendar and output controls.

## Initialization and presets

`Simulation.initialize(experiment, *, mesh=None)` validates before allocation.
Geometry in the new definition uses global physical cells without storage halos;
adapt into existing OceanGeometry initialization and preserve its metric
equations. The depth convention is the existing adapter's signed bottom value.
Prepare field-specific halos, masks, SeaIceLoad and regularized inverse ice
thickness for custom experiments; derive wSpeed from prescribed uWind/vWind
unless explicitly supplied. Input fields and initial ice override allocated
defaults, but geometry owns masks/metrics and derived fields remain derived.

`Experiment.builtin(name, ...)` provides island, dynamics and growth definitions
using their existing scenario registries and initializers. Preserve their exact
initial State, including historically default intermediate fields and dry-wall
ice values. Do not impose custom-experiment cleanup on legacy presets. Retain a
prescribed-ocean geometry example with no Veros dependency. Move reusable preset
construction behind the facade and make CLI entry points use its workflow.

Sharded allocation adapts global dimensions to local Configuration. Reuse
addressable-partition callbacks rather than materializing global arrays on every
rank. Pack each partition's halos and exchange inside shard_map.

## Execution and numerical compatibility

New pure `advance(state, configuration, constants, forcing, *, physics)` returns
State and Diagnostics for ONE timestep. Existing `veris.step` stays the public
MULTI-step rollout with unchanged behavior. Pure functions use no clocks,
mutable cursors, files, meshes or host conversions. Static arguments are
physics/configuration/constants; State and forcing remain dynamic AD leaves.

Preserve dynamics_transport order: mass, face concentration, wind forcing,
strength, momentum, pre-transport ocean stresses, advection, cleanup, ridging.
Then optionally Growth and final field-aware State/Diagnostics halo refresh.
Refresh prescribed forcing halos before kernels consume them. Keep the growth
preset's recursive Qnet/Qsw policy; island supplies Qnet=cooling and Qsw=0 before
every step. Custom explicitly prescribed heat fields are reapplied each step.
Spatial forcing uses field-aware sharding, rather than the old replicated-only
cooling specification. Preserve closed-y masks, normal flow and shear policies.

`Simulation.step()` advances one step; `run(steps=None, observe=None)` advances
an explicit count or the remaining configured duration. Both share the same
forcing cursor, absolute step count and exact integer microsecond model clock.
Return `RunResult(state, diagnostics, steps, elapsed, observations, timing)`;
observations are optional and post-step only. Successive run splits must equal
an uninterrupted run. Reject exhausted series and invalid counts before effects.

Reuse veris.step and extend scheduled execution to carry absolute step offsets
and pending device means across successive runs. No default trajectory history;
scheduled means retain float64 sums/counts and established half-open sampling.
Meshes, output manager, calendar and accumulators stay host resources, closed on
normal and exceptional exits. Save resolved metadata in histories and snapshots.

## Input, visualization and restart

Selected netCDF input updates initialized fields and refreshes appropriate
derived fields/halos; it is explicitly distinct from complete restart.
Visualization snapshots retain existing field selection/precision/halo removal.

Versioned restart restores all State fields, complete validated settings and
constants, global grid, physics/preset/forcing data and cursor, exact elapsed
microseconds and completed steps, calendar origin, boundary behavior and output
policy. Reconstruct through host allocation, then refresh field-aware halos.
Reject ordinary snapshots, means, partial/missing fields, mismatched dtypes,
incompatible dimensions, metadata or schema versions before execution.

Support same-backend serial CPU and single-device GPU restart first. Distributed
restart is explicitly rejected until repartitioned construction is implemented.
Resume to a NEW output file. Preserve absolute sampling/calendar origin, discard
unfinished pre-restart means and reject writing the crossing incomplete window;
only fully covered subsequent windows are emitted. Document this difference from
uninterrupted output while State trajectories remain equivalent.

## Verification and delivery

Test new behavior before implementation, one pytest instance at a time. Preserve
all existing passing tests and tolerances. Prove all-field preset equivalence,
custom geometry/masks/initial conditions, forcing and time across run splits,
complete save/resume, JIT and initial/forcing JVP/VJP/FD, serial/sharded boundaries
and scheduled output, invalid input errors, resource closure and import purity.
Run fast checks during development; full CPU correctness/coverage before each
code commit, maintained Ruff/format/annotations/ty, clean built installation and
public imports, and Sphinx warnings-as-errors. Report unavailable GPU/multihost
checks honestly. Update CHANGELOG after milestones; commits target jax-only.
Deliver a primary public quickstart, API reference, migrated example CLIs,
migration guide and requirement-by-requirement completion evidence.
