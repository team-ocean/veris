Simulation lifecycle
====================

``Simulation`` is the host interface for a validated experiment. It reuses
``initialize_experiment``, the single physics timestep ``advance`` and the
existing ``step`` rollout. Numerical State contains arrays; clocks, execution
resources and optional histories belong to the host facade. The
:doc:`/quickstart/user-guide` provides the complete run/output/restart example.

``run()`` advances the remaining configured duration; ``run(n)`` explicitly
advances n timesteps, including beyond that default target if forcing and the
clock permit. ``step()`` advances exactly one timestep. ``run(0)`` returns the
current fields and time without tracing physics when no observer is supplied.
Model time uses the active timestep at integer microsecond precision.

Forcing series use the absolute completed-step position: split runs select the
same samples as one uninterrupted run. Series override constants with the same
field name. Missing fields retain the numerical State's current values;
prescribed heat is reapplied before each step even when thermodynamics overwrites
it with a coupling flux. See :doc:`experiment` for shapes, units and precision.

``RunResult`` contains the final ``state`` and ``diagnostics``, this call's
``steps``, absolute ``completed_steps`` and ``elapsed``, optional
``observations`` and wall-clock ``compilation_seconds`` and
``integration_seconds``. Diagnostics is ``None`` until the first successful
physics timestep. Histories are omitted by default. A pure observer selects
post-step values and returns a PyTree whose leaves stack along the time axis;
see the observation example in :doc:`/quickstart/user-guide`.

An observer must use pure JAX operations and have fixed output shapes. A zero
run with an observer traces that observer to infer empty histories. Invalid run
counts, exhausted forcing and clock overflow fail before execution, leaving the
simulation open and unchanged. Tracing or device failures close it and preserve
its last committed State and time. ``close()`` is idempotent; context-manager
exit closes on success or failure.

Initialize distributed JAX and choose backend settings before constructing a
simulation. The facade never changes process-wide JAX configuration. A supplied
mesh is borrowed; its lifetime remains the caller's responsibility. Supported
initialization modes are documented in :doc:`experiment`.

The mutable facade is host-only. Use :func:`veris.advance` and
:func:`veris.step` directly for JIT, gradients, JVP and VJP; see
:doc:`/reference/integration` for the pure execution contract.

Scheduled output and selected input
-----------------------------------

Define ``OutputSettings`` separately from numerical settings, attach it to the
Experiment and supply ``output_path`` at initialization. The worked example in
:doc:`/quickstart/user-guide` combines streams with successive runs.

The writer is lazy and refuses an existing destination. Output-disabled or
unconfigured experiments need no output path. Streams retain the original
calendar origin and absolute sampling cadence across calls. A run split inside
a mean window retains its sums and sample
count; closing discards an unfinished window. Sampling, precision, halo removal
and calendar policies are described in :doc:`/reference/io`.

Output and pure ``observe`` histories may be requested together on the same
trajectory. Default scheduled output retains only running sums/counts and has no
timestep history. Selected observations necessarily occupy memory proportional
to their number of steps and selected values; transient history chunks default
to eight timesteps. Resolved experiment, numerical settings and constants are
saved as file attributes, preserving reproducible definitions.

``snapshot`` writes an instantaneous selected-field visualization file using
the established ``snapshot`` group and absolute model time. It does not flush
unfinished mean windows. ``load_fields(path, variables=None, record=-1,
group="snapshot")`` loads an instantaneous record into an initialized serial
simulation. It supports the initial-condition and prescribed forcing names from
:doc:`experiment`; geometry, masks and independent derived metrics require
reinitialization. Mean records, incompatible shapes, nonfinite values, invalid
physical ranges and unsafe precision conversions fail before State changes.
Representable float64 input can round into a float32 model. Loading refreshes
ice load/inverse thickness, derived wind speed and field-specific boundary halos,
and honors dry cells and blocked ice velocity faces. It retains the model clock,
forcing definition/cursor, pending means and last timestep's Diagnostics; the
next timestep reapplies its prescribed inputs. Neither selected input nor a
visualization snapshot is a complete restart.

.. _complete-restart:

Complete serial restart
-----------------------

``simulation.save_restart(path)`` publishes a new complete checkpoint without
overwriting an existing file. It leaves the live numerical State, forcing
position and pending output means unchanged. ``Simulation.from_restart``
validates and restores the saved experiment without running an initializer or
reapplying startup forcing. The :doc:`/quickstart/user-guide` checks a resumed
trajectory against uninterrupted execution.

Version 1 supports serial CPU and a single GPU device on the actual saved
backend. It rejects distributed partitions, multi-device/non-addressable State,
missing fields, incompatible precision, corrupted dimensions/metadata, unsupported
versions and visualization snapshots. Restoring onto another backend is not a
supported migration. GPU restoration requires an available compatible GPU;
the package never silently falls back or changes JAX configuration.

Checkpoints contain every native State array, including complete halos and the
physical closed-y north-wall shear traction. They retain resolved numerical
configuration, physical constants, the complete experiment and typed input
arrays, future/past forcing, its absolute position, completed steps and exact
integer-microsecond model time. Validation finishes on the host before backend
enumeration or device placement. Saved halos are restored verbatim because
cyclic thermodynamics retains reference-significant evolved halo values; normal
boundary operations resume with the next timestep. Diagnostics are ephemeral
step results and reset to ``None`` until that timestep.

Enabled output requires a new destination. Original calendar, origin and absolute
cadence remain. Optional ``output_settings=...`` may replace streams/policies
while retaining the same origin/calendar. The checkpoint instant is skipped,
including a time-zero checkpoint. Unfinished old mean sums are discarded;
windows crossing the saved time are omitted, and only fully covered resumed
windows are written. Snapshot/input helpers remain selected-field interfaces.
No-output model clocks use the full supported ``timedelta`` range independently
of calendar date limits; enabled calendar output validates its date range.

The netCDF schema marks ``veris_file_kind="restart"`` and
``veris_restart_version=1``. Its ``state`` group holds one complete native
halo-inclusive record; ``definition`` groups hold typed geometry, initial and
forcing arrays with validated references. Exact JSON integer clock metadata is
authoritative; redundant CF seconds support ordinary netCDF tooling. Files are
written and closed under a temporary name, then published atomically without
clobbering. Incompatible files fail clearly rather than allocating missing-field
defaults. Hard-link publication requires filesystem support.

.. autoclass:: veris.execution.simulation.Simulation
   :members:

.. autoclass:: veris.execution.simulation.RunResult
   :members:

Public experiment runner
------------------------

``run_experiment`` accepts the same Experiment definition and grouped overrides
as the file codec. It preflights inputs, calendars and output destinations,
prepares runtime resources, executes Simulation and returns its synchronized
RunResult. ``execution_metadata`` records the actual backend, partition shape,
process count and process index; snapshot/history attributes retain those values
separately from the experiment definition.

A supplied ``output_path`` enables the authored sampling schedule. When the
experiment has no schedule, the runner uses one instantaneous sample per step.
``snapshot_path`` writes final physical cells. Missing output parents are created
after validation; existing files remain protected. Imports and definition
inspection neither change JAX policy nor select devices. Runtime precision and
backend policy are applied by the explicit launch operation.

.. autofunction:: veris.run_experiment
