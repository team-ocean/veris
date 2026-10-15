Scan rollouts and checkpointing
===============================

``veris.step`` repeats a fixed number of model timesteps using
`jax.lax.scan <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>`_.
A timestep advances the model by one time interval. ``veris.advance`` performs
one complete physics timestep. The shared rollout driver repeats it; a rollout
is this sequence of steps. Use ``functools.partial`` to fix Configuration,
PhysicalConstants and the physics mode for the run. The default result is the
final State, the collection of model arrays, without a saved State history::

   from functools import partial
   import jax
   jax.config.update("jax_enable_x64", True)

   from veris import Experiment, advance, initialize_experiment, step

   experiment = Experiment.builtin("island")
   initial, settings, constants = initialize_experiment(experiment)
   transition = partial(
       advance, configuration=settings, constants=constants,
       forcing={"Qnet": 100.0, "Qsw": 0.0}, physics=experiment.physics,
   )
   final = step(initial, transition, 3, has_aux=True)

``steps`` is a nonnegative integer fixed during compilation. The transition
must be pure (no changes to outside resources) and traceable by JAX (JAX can
inspect its operations). The carry, or data passed from step to step, must keep
the same structure, array shapes and dtypes (number types). Reuse the same
functions with fixed arguments across rollouts to reuse compiled scans.
Settings and physical constants keep their existing fixed treatment. State
and forcing arrays are differentiable: JAX can calculate how results change
when these inputs change. Keep changes to Python objects, NumPy conversions
and file output outside the transition and observer. ``advance`` returns
``(State, Diagnostics)``. ``has_aux=True`` passes the updated State to the next
step as the rollout carry.

Forcing fields are prescribed inputs, such as wind or heat, and are reapplied
before each timestep. You can supply scalars or physical ``(nx, ny)`` JAX
arrays. Each field gets its own halo preparation; halos are extra cells that
hold boundary or neighboring data. Supplying uWind or vWind calculates wSpeed
unless it is explicitly prescribed. Missing fields keep their current State
values, including growth Qnet/Qsw carried from the previous step. Host
Experiment definitions check finite values and units outside JAX operations.
Checks for initial ice and prescribed heat/wind sensitivities cover JIT
(compilation), gradients, JVP (forward directional derivatives), VJP (reverse
derivatives) and finite differences (small input changes). Fixed settings,
constants and host definitions stay outside differentiation. Physical
threshold choices remain those of :doc:`automatic-differentiation`.

Physics modes
-------------

``advance`` selects the complete physics sequence through ``physics``:

* ``"coupled"`` advances dynamics (ice motion), transport (movement between
  cells) and growth (freezing and melting). Prescribed heat supplied through
  ``forcing`` is refreshed before each timestep.
* ``"dynamics"`` advances momentum and transport with prescribed ocean and wind
  fields.
* ``"thermodynamics"`` advances growth. Missing ``Qnet`` and ``Qsw`` keep the
  values returned by the preceding step, as in the reference column.

The core ``veris.physics.dynamics.driver.dynamics_transport`` stage advances
momentum, calculates ocean stress before transport, then applies advection
(movement with ice velocity), cleanup and ridging (ice piling up). It returns
``(state, stress_u, stress_v)`` with arrays that include halos. Specialized
combinations add forcing, optional growth, the context for splitting arrays
across devices, and final State/Diagnostics halos. When arrays are split this
way, this local stage runs inside the caller's ``jax.shard_map``.

For example, a growth rollout uses the same public transition::

   experiment = Experiment.builtin("growth")
   initial, settings, constants = initialize_experiment(experiment)
   transition = partial(
       advance, configuration=settings, constants=constants,
       physics=experiment.physics,
   )
   final = step(initial, transition, 3, has_aux=True)

Selected observations and diagnostics
-------------------------------------

A pure observer selects an array PyTree, a nested collection of arrays, after
each State update. For the ``advance`` transition, it accepts State and
Diagnostics. The result is ``(final_state, history)``. Each array in the history
has a first axis for time, with length ``steps``. The history excludes the
initial State::

   final, history = step(
       initial, transition, 3, has_aux=True,
       observe=lambda state, diagnostic: {"ice": state.hIceMean[2:-2, 2:-2]},
   )

For output-only coupling fields, use ``has_aux=True`` with a transition returning
``(state, diagnostics)`` and an observer accepting both arguments::

   final, history = step(
       initial, transition, 3, has_aux=True,
       observe=lambda state, diagnostic: {
           "ice": state.hIceMean,
           "freshwater": diagnostic.EmPmR,
       },
   )

Only State passes from step to step. Diagnostics are discarded when no
observer is supplied. An observer can keep diagnostics calculated partway
through the physics sequence, such as ocean stress before transport and growth.

Zero steps keep the initial State and return empty histories with the expected
shapes. JAX still inspects the transition and observer to determine those
shapes, including in auxiliary-output mode. A callback that JAX cannot trace
is still invalid when the step count is zero. Observe the initial State
separately if needed.

Time-varying forcing and AD
---------------------------

Automatic differentiation (AD) calculates how a result changes with its
inputs. ``inputs`` accepts a nonempty array PyTree where every array has a
first axis of length ``steps``. The transition then receives
``transition(state, input_slice)``. For example, calculate how a thickness
objective changes with each prescribed cooling value::

   import jax
   import jax.numpy as jnp
   jax.config.update("jax_enable_x64", True)
   from veris import Experiment, advance as physics_timestep, initialize_experiment, step

   experiment = Experiment.builtin("island")
   initial, settings, constants = initialize_experiment(experiment)
   def transition(state, prescribed):
       return physics_timestep(
           state, settings, constants, prescribed, physics=experiment.physics,
       )
   cooling = jnp.asarray([80.0, 100.0, 120.0], dtype=initial.hIceMean.dtype)

   def objective(forcing):
       final = step(
           initial, transition, 3,
           inputs={"Qnet": forcing, "Qsw": jnp.zeros_like(forcing)},
           has_aux=True, checkpoint=True,
       )
       return final.hIceMean[2:-2, 2:-2].mean()

   sensitivity = jax.grad(objective)(cooling)

``checkpoint=True`` is the default. It wraps the repeated step calculation,
including the observer, with ``jax.checkpoint``. Reverse AD recalculates
intermediate step values so it needs to save fewer values, at the cost of more
computation. This does **not** guarantee constant memory as a rollout grows
longer. Saved carry values and requested observation histories can still grow
with the number of timesteps. Use ``checkpoint=False`` to compare the ordinary
scan. Multi-process rollouts retain halo exchange results for reverse AD rather
than exchanging those values again. Arithmetic is still recalculated; saved
halo values add storage at each timestep and inner solver iteration. See JAX's
`gradient checkpointing guide <https://docs.jax.dev/en/latest/gradient-checkpointing.html>`_
and :doc:`automatic-differentiation` for physical derivative conventions.

Sharded rollouts
----------------

A mesh arranges devices along the grid axes. Initialize it and keep its
context active around the rollout and its JAX operations. Define global
physical dimensions and matching partitions. Regional initialization creates
the packed State, which includes each partition's halos::

   import jax
   from veris import Execution, Experiment, Grid, advance, initialize_experiment, step

   layout = (1, jax.device_count())
   mesh = jax.make_mesh(layout, ("x", "y"))
   experiment = Experiment.builtin(
       "dynamics", grid=Grid(8, 8 * layout[1]),
       execution=Execution(partitions=layout),
   )
   with jax.set_mesh(mesh):
       initial, settings, constants = initialize_experiment(experiment, mesh=mesh)
       transition = partial(
           advance, configuration=settings, constants=constants,
           physics=experiment.physics,
       )
       final = step(initial, transition, 3, has_aux=True)

Each partition keeps its own halos. Selected history arrays gain a first axis
for time that is not split across devices. Select one time slice before using
the spatial output collector, which gathers grid fields. Coupled presets and
custom OceanGeometry use the same rules for global physical inputs. Keep
``jax.set_mesh(mesh)`` active around pure advance/step and their JAX operations.
Public advance accepts scalars or forcing arrays on the global physical grid.
Forcing series keep a first axis for time that is not split across devices.
See :doc:`/api/experiment`.

Host output and timing
----------------------

The dynamics, growth and parallel command-line runners use Simulation, which
calls ``veris.execution.rollout_output.run_timed``. You can also use the
low-level timing API directly. It prepares and compiles each required scan
shape with ``jax.jit(...).lower(...).compile()`` before running the model.
Compilation can inspect pure functions, but it executes no model transitions,
device callbacks, sample collection or file writes. The returned tuple
contains final State, compilation seconds and integration seconds. Integration
timing includes execution, waiting for device results and host output.

Use ``output_callbacks(manager, step_seconds)`` to connect an OutputManager::

   from veris.execution.rollout_output import output_callbacks, run_timed

   def state_transition(state):
       return transition(state)[0]

   observe, select = output_callbacks(manager, experiment.step_duration.total_seconds())
   final, compilation_seconds, integration_seconds = run_timed(
       initial, state_transition, 100, observe=observe, select=select,
   )

The :ref:`scheduled-output-integration` reference defines where chunks of
steps end, how means are accumulated, what collectors must do, and how to use
a manager dedicated to writing reduced fields. It also describes calendar and
sampling rules alongside direct ``OutputManager.sample`` usage.

Custom host observers keep the existing ``observe(sample, iteration)``
interface. They receive selected initial and later samples from histories of
limited length. The default chunk cap is eight; ``chunk_size`` changes that
cap. With no observer, including when OutputManager output is disabled, the
runner compiles and executes one full-length scan, saves no observation
histories and ignores the chunk cap. Zero-step host runs neither compile nor
execute physics, but can write a requested initial instantaneous record. Final
snapshots remain separate host operations. Use ``veris.step`` directly for AD.
The timing/output wrapper runs only on the host.

See :doc:`/api/integration` for Python signatures and docstrings.
