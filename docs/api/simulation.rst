Simulation lifecycle
====================

``Simulation`` controls a checked experiment from Python on the host, the
computer that manages the calculation. It reuses ``initialize_experiment``,
``advance`` for a single physics timestep, and ``step`` to repeat timesteps.
Numerical State contains arrays of model values. The host interface holds
clocks, execution resources and optional histories of step results. The
:doc:`/quickstart/user-guide` provides the complete run/output/restart example.

``run()`` advances through the remaining configured duration. ``run(n)``
advances n timesteps, including beyond that default target if the supplied
external inputs and clock allow it. ``step()`` advances exactly one timestep.
``run(0)`` returns the current fields and time without tracing physics when no
observer is supplied. Tracing means JAX examines a function to prepare its
calculation. Model time uses the active timestep at integer microsecond precision.

Forcing means prescribed external inputs such as wind or ocean temperature.
Forcing series use the total number of completed steps to choose their position.
Split runs therefore select the same samples as one uninterrupted run. Series
override constants with the same field name. Missing fields keep the numerical
State's current values. Prescribed heat is reapplied before each step even when
thermodynamics, the freezing and melting calculation, overwrites it with a heat
flux for exchange with another model. See :doc:`experiment` for shapes, units
and precision.

``RunResult`` contains the final ``state`` and ``diagnostics``, this call's
``steps``, total ``completed_steps`` and ``elapsed`` since the model's start,
optional ``observations`` and real elapsed ``compilation_seconds`` and
``integration_seconds``. Diagnostics holds step results and is ``None`` until
the first successful physics timestep. Histories are omitted by default. An
observer selects values after each step without Python side effects. It returns
a PyTree, a nested collection of data, whose arrays stack along the time axis.
See the observation example in :doc:`/quickstart/user-guide`.

An observer must use JAX operations without Python side effects and return fixed
output shapes. A zero run with an observer traces that observer to determine
the shapes of empty histories. Invalid run counts, exhausted forcing and clock
overflow fail before execution, leaving the simulation open and unchanged.
Tracing or device failures close it and preserve the last State and time saved
in the host interface. Calling ``close()`` repeatedly has the same effect as
calling it once. Leaving a context manager closes the simulation on success or
failure.

Initialize JAX's services for execution across processes and choose backend
settings before constructing a simulation. The backend is the CPU or GPU system
that runs JAX. This interface never changes process-wide JAX configuration. A
supplied mesh arranges the devices that share the grid; the caller remains
responsible for its lifetime. Supported initialization modes are described in
:doc:`experiment`.

This interface changes Python objects and runs only on the host. Use
:func:`veris.advance` and :func:`veris.step` directly for JIT compilation,
gradients, JVP and VJP. JIT prepares calculations to run on a device. Gradients
measure how results change with inputs. JVP follows a chosen input change
forward; VJP carries output sensitivities back to inputs. See
:doc:`/reference/integration` for execution without Python side effects.

Scheduled output and selected input
-----------------------------------

Define ``OutputSettings`` separately from numerical settings, attach it to the
Experiment and supply ``output_path`` at initialization. A stream selects fields
and a sampling schedule. The worked example in :doc:`/quickstart/user-guide`
combines streams with successive runs.

The writer opens a file only when needed and refuses an existing destination.
Experiments with output disabled or no output configuration need no output path.
Streams keep the original calendar start and absolute sampling schedule across
calls.
A mean window is an interval over which samples are averaged. Splitting a run
inside a mean window keeps its sums and sample count; closing discards an
unfinished window. :doc:`/reference/io` describes sampling, precision, calendars
and removal of halos, the extra cells holding boundary or neighboring values.

You can request output and ``observe`` histories from the same sequence of model
steps. The observer runs without Python side effects. Default scheduled output
keeps only running sums/counts and no timestep history. Observations need memory in
proportion to their number of steps and selected values. Temporary history
chunks default to eight timesteps. The file attributes save the resolved
experiment, numerical settings and constants so the definitions are reproducible.

``snapshot`` writes selected fields at a single instant for visualization, using
the established ``snapshot`` group and absolute model time. It does not write
out unfinished mean windows. ``load_fields(path, variables=None, record=-1,
group="snapshot")`` loads an instantaneous record into an initialized serial
simulation. It supports the initial-condition and prescribed forcing names from
:doc:`experiment`. Geometry, masks and independent derived grid distances and
areas require reinitialization. Mean records, incompatible shapes, infinities
or undefined numerical values, invalid physical ranges and unsafe precision
conversions fail before State changes. Representable float64 input can round
into a float32 model. Loading refreshes ice load/inverse thickness, computed
wind speed and each field's boundary halos. It respects dry cells and blocked
cell faces for ice velocity. It keeps the model clock, forcing definition and
position tracker, pending means and the last timestep's Diagnostics. The next
timestep reapplies its prescribed inputs. Selected input and visualization
snapshots do not contain everything needed for a complete restart.

.. _complete-restart:

Complete serial restart
-----------------------

``simulation.save_restart(path)`` saves a new complete checkpoint, a file with
everything needed to resume the run, without overwriting an existing file. It
leaves the live numerical State, forcing position and pending output means
unchanged. ``Simulation.from_restart`` checks and restores the saved experiment
without running an initializer or reapplying startup forcing. The
:doc:`/quickstart/user-guide` checks that a resumed run matches uninterrupted
execution.

Version 1 supports serial CPU and a single GPU device on the backend saved in
the file. It rejects distributed partitions, State spread across multiple
devices or inaccessible to the current process, missing fields, incompatible
precision, corrupted dimensions or metadata, unsupported versions and
visualization snapshots. Restoring onto another backend is not supported.
GPU restoration requires an available compatible GPU. Veris never silently
falls back to another backend or changes JAX configuration.

Checkpoints contain every native State array, including complete halos and shear
traction, the tangential force per unit area, at the physical north wall when
the y boundary is closed. They keep the resolved numerical configuration,
physical constants, complete experiment and input arrays with their data types.
They also keep future/past forcing, its absolute position, completed steps and
exact integer-microsecond model time. Checks finish on the host before listing
backend devices or placing arrays on them. Saved halos are restored exactly:
thermodynamics with periodic boundaries, whose opposite ends join, retains
evolved halo values that matter for agreement with the reference code. Normal
boundary operations resume with the next timestep. Diagnostics are temporary
step results and reset to ``None``
until that timestep.

Enabled output requires a new destination. The original calendar, starting date
and absolute sampling schedule remain. Optional ``output_settings=...`` may
replace streams/policies while keeping the same origin/calendar. Output skips
the checkpoint instant, including a time-zero checkpoint. Unfinished old mean
sums are discarded. Windows that cross the saved time are omitted, and only
windows fully covered by the resumed run are written. Snapshot/input helpers
still handle selected fields. Without output, model clocks use the full
supported ``timedelta`` range independently of calendar date limits. Enabled
calendar output checks its date range.

The netCDF file format marks ``veris_file_kind="restart"`` and
``veris_restart_version=1``. Its ``state`` group holds one complete native
record including halos. The ``definition`` groups hold geometry, initial and
forcing arrays with their data types and checked references. Exact JSON integer
clock metadata is the source of truth. Additional seconds following CF, the
Climate and Forecast conventions, let ordinary netCDF tools read the time.
Veris writes and closes a file under a temporary name, then publishes it in
a single filesystem operation without overwriting an existing file. Incompatible
files produce clear errors rather than filling missing fields with defaults.
Publication uses a hard link, so the filesystem must support hard links.

.. autoclass:: veris.execution.simulation.Simulation
   :members:

.. autoclass:: veris.execution.simulation.RunResult
   :members:

Public experiment runner
------------------------

``run_experiment`` accepts the same Experiment definition and grouped overrides
as the file reader. It checks inputs, calendars and output destinations before
execution, prepares resources, runs Simulation and returns RunResult after
the device calculations finish. ``execution_metadata`` records the actual
backend, partition shape, process count and process index. Snapshot/history
attributes keep those values separately from the experiment definition.

A supplied ``output_path`` enables the sampling schedule in the definition.
When the experiment has no schedule, the runner uses one instantaneous sample
per step. ``snapshot_path`` writes the final physical cells. Missing parent
directories for output are created after validation; existing files remain
protected. Importing modules and inspecting definitions do not change JAX policy
or select devices. The explicit launch operation applies runtime precision and
backend policy.

.. autofunction:: veris.run_experiment
