Migrating existing Veris experiments
====================================

Existing pure rollouts and setup compatibility entry points retain their
contracts. New applications can use ``Experiment`` and ``Simulation`` to share
initialization, forcing, time accounting and output across physics modes.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Existing entry point
     - Centralized workflow
   * - ``setups.island.initialize`` / dynamics / growth factories
     - ``Experiment.builtin(...)`` and ``Simulation.initialize(...)``
   * - Setup ``step`` / ``step_with_diagnostics``
     - ``advance(...)`` for one pure timestep, or ``simulation.step()``
   * - ``veris.step(state, transition, n)``
     - Unchanged pure rollout; ``simulation.run(n)`` adds host lifecycle/time
   * - ``initialization.initialize`` with packed overrides
     - Low-level allocation remains; ``initialize_experiment`` creates runnable
       preset or custom geometry through physical global inputs
   * - ``write_snapshot`` / ``save_final``
     - ``simulation.snapshot(...)`` at the current absolute model time
   * - ``read_record`` / ``update_state``
     - ``simulation.load_fields(...)`` for validated selected serial input
   * - A selected-field snapshot used as a restart
     - ``save_restart`` / ``from_restart`` for complete supported serial state

The distinction between timestep and rollout is explicit: ``advance`` performs
one physics step and returns ``(State, Diagnostics)``; the established public
``step`` performs n transitions with a supplied callable. ``RunResult`` wraps
the host run's final arrays, time/counters, optional observations and timings.
Keep JAX losses on ``advance`` and ``step`` rather than mutable Simulation.

Dimensions and configuration
----------------------------

``Grid(nx, ny)`` counts global physical cells, with no halos. Low-level
``Configuration.nx``/``ny`` and the allocator's dimensions remain per partition.
For example, Grid(12, 8) with Execution(partitions=(3, 2)) produces local (4, 4)
cells and packed State storage (24, 16). Each partition stores two halo cells
on each side. Supply custom geometry/initial/forcing spatial inputs in global
``(nx, ny)`` physical shape; the public initializer packs them regionally.

Use ``settings`` for physics/solver/precision controls, ``constants`` for
material parameters, ``scenario`` for builtin controls, ``initial_conditions``
and ``forcing`` for experiment fields, ``execution`` for backend/partitions
and ``output`` for host streams/calendar. Grid, dtype and duration do not also
belong in scenario. Numerical defaults and validation continue to come from
the existing registries. Declarative JSON applies registry/preset defaults,
file values, then explicit Python overrides. Arrays replace whole values.

Allocation defaults do not produce a custom physical experiment. Supply
``OceanGeometry`` through the public definition or choose a builtin.
Initialization establishes masks, metrics, derived quantities and boundaries;
see :doc:`custom-ocean`. User experiments require no source edits.

Forcing and successive runs
---------------------------

Constants are scalars or global fields; series have a leading time dimension.
The absolute completed-step index chooses a series sample before each physics
step. Series override constants by name, and absent fields retain their current
State values. This matters for growth's recursive heat policy: copying its
initial Qnet into constant forcing changes the experiment. The island preset
prescribes heat each step because growth overwrites Qnet/Qsw with coupling fluxes.

``simulation.run(2); simulation.run(3)`` follows the same forcing/time sequence
as ``run(5)``. ``run()`` uses the remaining configured target, while explicit
counts may exceed it when forcing coverage and clock bounds permit. Failed
preflight leaves the simulation usable; execution failures close resources and
retain the last committed numerical State/time. Context managers close on
success or failure.

Output, selected input and complete restart
-------------------------------------------

Scheduled output keeps established precision, calendar and halo-removal rules.
Instant records use model precision at post-step endpoints. Means use float64
pre-step samples over half-open windows; exact calendar bounds may fall between
timesteps. Successive runs retain unfinished sums/counts. Closing discards
unfinished means. Optional pure histories share the trajectory and retain only
selected observations; default output stores no timestep history.

``load_fields`` accepts supported initial/forcing fields from instantaneous
serial records, validates the whole selection before committing, refreshes
derived quantities/halos, and retains time, prescribed forcing and pending means.
Specify fields explicitly when reading a full visualization snapshot; geometry,
masks and independent metrics require reinitialization. Selected input may round
representable float64 data into float32 and is not an exact restart.

Version 1 complete restart preserves all native State arrays including halos,
resolved configuration/constants/definition, complete forcing data and position,
and integer-microsecond model time. It supports serial CPU or one GPU device on
the saved backend. Incomplete/incompatible data and distributed State are
rejected. No startup forcing or initializer is reapplied. Diagnostics reset to
None until a new timestep. Saving atomically publishes a new file and leaves
the live simulation/output untouched.

Restore into a new output file. The original origin/calendar and absolute cadence
remain; stream settings may be replaced without changing that origin/calendar.
Checkpoint instants are skipped, including time zero. Old unfinished/crossing
mean windows are discarded, and only windows fully covered after the checkpoint
are written. Resume and uninterrupted numerical trajectories match; their output
files differ intentionally at these omitted restart windows.

Execution resources and limitations
-----------------------------------

Select JAX backend/precision before device initialization. Initialize distributed
JAX explicitly, construct the x/y mesh, then initialize/run the experiment.
Simulation borrows mesh resources; caller-owned distributed services remain
caller-owned. Maintained CLI runners use this lifecycle while retaining options,
outputs and bootstrap ownership. Importing Veris or a setup has no backend side
effect.

Settings/constants, definition construction and host I/O are static/outside AD.
Initial State and supplied JAX forcing retain tested JIT, grad, JVP and VJP paths;
physical branch thresholds retain their existing derivative conventions. GPU
and actual multi-host execution need compatible hardware/runtime; local CPU
sharding tests do not establish those hardware checks. Distributed restart is
unsupported. See :doc:`/api/simulation` and
:doc:`/reference/automatic-differentiation` for the precise contracts.
