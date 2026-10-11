Simulation lifecycle
====================

``Simulation`` is the host interface for a validated experiment. It reuses
``initialize_experiment``, the single physics timestep ``advance`` and the
existing ``step`` rollout. Numerical State contains arrays; clocks, execution
resources and optional histories belong to the host facade::

   from veris import Experiment, Grid, Simulation

   experiment = Experiment.builtin(
       "island", grid=Grid(6, 8),
       settings={"dtype": "float32", "nEVPsteps": 2}, steps=5,
   )
   with Simulation.initialize(experiment) as simulation:
       first = simulation.run(2)
       result = simulation.run()
       assert first.steps == 2
       assert result.steps == 3
       assert result.completed_steps == 5
       assert result.elapsed == 5 * experiment.step_duration
       final_state = result.state

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
post-step values and returns a PyTree whose leaves stack along the time axis::

   import jax.numpy as jnp

   with Simulation.initialize(experiment) as simulation:
       result = simulation.run(
           observe=lambda state, diagnostics: {
               "ice": jnp.mean(state.hIceMean[2:-2, 2:-2]),
               "freshwater": jnp.mean(diagnostics.EmPmR[2:-2, 2:-2]),
           },
       )
       assert result.observations["ice"].shape == (5,)

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

Define output streams separately from numerical settings and provide a new
host destination at initialization::

   from datetime import timedelta
   from veris.io import OutputSettings, Stream, read_record

   output = OutputSettings(streams=(
       Stream("instant", ("hIceMean", "Area"), timedelta(seconds=600)),
       Stream("mean", ("hIceMean",), timedelta(seconds=600),
              timedelta(seconds=1200)),
   ))
   experiment = Experiment.builtin(
       "island", grid=Grid(6, 8), steps=5,
       settings={"dtype": "float32", "nEVPsteps": 2}, output=output,
   )
   with Simulation.initialize(experiment, output_path="history.nc") as simulation:
       simulation.run(2)
       result = simulation.run()
       simulation.snapshot("final.nc", variables=("hIceMean", "Area"))
   record = read_record("final.nc")
   assert record.time == result.elapsed.total_seconds()
   assert record.fields["hIceMean"].shape == (6, 8)

The writer is lazy and refuses an existing destination. Output-disabled or
unconfigured experiments need no output path. Streams retain the original
calendar origin and absolute sampling cadence across calls. Instantaneous
samples occur at post-step endpoints, with the optional initial sample emitted
once. Arithmetic means sample pre-step fields over half-open windows and
accumulate in float64; point records retain model precision. A run split inside
a mean window retains its sums and sample count. Closing discards an unfinished
window. Two partition halos are removed through the existing serial or
distributed collector at writing time.

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

.. autoclass:: veris.simulation.Simulation
   :members:

.. autoclass:: veris.simulation.RunResult
   :members:
