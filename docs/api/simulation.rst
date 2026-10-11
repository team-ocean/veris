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

.. autoclass:: veris.simulation.Simulation
   :members:

.. autoclass:: veris.simulation.RunResult
   :members:
