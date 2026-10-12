Scan rollouts and checkpointing
===============================

``veris.step`` advances a fixed number of model timesteps using
`jax.lax.scan <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>`_.
``veris.advance`` supplies one complete physics timestep; the shared rollout
driver supplies time iteration. Bind Configuration, PhysicalConstants and the
physics mode with ``functools.partial``.
The default result is the final State, without an explicit State history::

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

``steps`` is a static nonnegative integer. The transition must be pure and JAX
traceable: carry structure, array shapes and dtypes must stay fixed. Reuse bound
callables across rollouts to reuse compiled scans. Settings and physical
constants keep their existing static treatment; State and forcing arrays are
differentiable. Python mutation, NumPy conversions and file output belong
outside the transition and observer. ``advance`` returns ``(State, Diagnostics)``;
``has_aux=True`` keeps the updated State as the rollout carry.

Supplied forcing fields are reapplied before each timestep. Scalars and physical
``(nx, ny)`` JAX arrays are supported, with field-specific halo preparation.
Supplying uWind or vWind derives wSpeed unless it is explicitly prescribed.
Absent fields retain their current State values, including recursive growth
Qnet/Qsw. Host Experiment definitions validate finite values and units outside
JAX transforms. Initial ice and prescribed heat/wind sensitivities have JIT,
gradient, JVP, VJP and finite-difference checks; static settings, constants and
host definitions are outside the differentiated path. Physical threshold
conventions remain those of :doc:`automatic-differentiation`.

Physics modes
-------------

``advance`` selects the complete physics sequence through ``physics``:

* ``"coupled"`` advances dynamics, transport and growth. Prescribed heat supplied
  through ``forcing`` is refreshed before each timestep.
* ``"dynamics"`` advances momentum and transport with prescribed ocean and wind
  fields.
* ``"thermodynamics"`` advances growth. Missing ``Qnet`` and ``Qsw`` retain their
  recursive values from the preceding step, as in the reference column.

The core ``veris.physics.dynamics.driver.dynamics_transport`` stage advances
momentum, computes ocean stress before transport, then applies advection,
cleanup and ridging. It returns ``(state, stress_u, stress_v)`` with halo-inclusive
arrays. Specialized compositions supply forcing, optional growth, the sharding
context and final State/Diagnostics halos. Under sharding, this local stage runs
inside the caller's ``jax.shard_map``.

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

A pure observer selects an array PyTree after each updated State. For the
``advance`` transition it accepts State and Diagnostics. The result is
``(final_state, history)``; each history leaf has a leading time axis of length
``steps``, excluding the initial State::

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

Only State remains in the carry. Diagnostics are discarded when no observer is
supplied. This preserves diagnostics evaluated at intermediate points of the
physics sequence, such as ocean stress before transport and growth.

Zero steps preserve the initial State and return empty histories with the
inferred shapes. JAX still traces the transition and observer to infer those
shapes, including in auxiliary-output mode; zero steps do not make an otherwise
untraceable callback valid. Observe the initial State separately if needed.

Time-varying forcing and AD
---------------------------

``inputs`` accepts a nonempty array PyTree whose every leaf has leading length
``steps``. The transition then receives ``transition(state, input_slice)``. For
example, differentiate a thickness objective with respect to each prescribed
cooling value::

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

``checkpoint=True`` is the default. It wraps the scan body, including the
observer, with ``jax.checkpoint``. Reverse AD recomputes step intermediates to
reduce saved residuals, at the cost of extra computation. This does **not**
guarantee constant memory with rollout length: scan carry residuals and requested
observation histories can still scale with the number of timesteps. Use
``checkpoint=False`` to compare the ordinary scan. See JAX's
`gradient checkpointing guide <https://docs.jax.dev/en/latest/gradient-checkpointing.html>`_
and :doc:`automatic-differentiation` for physical derivative conventions.

Sharded rollouts
----------------

Initialize the mesh and retain its context around the rollout and its JAX
transformations. Define global physical dimensions and matching partitions;
regional initialization constructs the packed State::

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

Each partition keeps its own halos. Selected history arrays gain an unpartitioned
leading time axis; select one time slice before using the spatial output
collector. Coupled presets and custom OceanGeometry use the same global physical
input contract. Retain ``jax.set_mesh(mesh)`` around pure advance/step and their
transformations. Public advance accepts scalar or global physical spatial
forcing; series retain an unpartitioned leading time axis. See
:doc:`/quickstart/user-guide` and :doc:`/api/experiment`.

Host output and timing
----------------------

The dynamics, growth and parallel command-line runners use Simulation, which
delegates to ``veris.execution.rollout_output.run_timed``. The low-level timing API
remains available. It lowers and compiles each required
scan shape with ``jax.jit(...).lower(...).compile()`` before executing the
trajectory. Compilation can trace pure callables, but executes no model
transitions, device callbacks, sample collection or file writes. The returned
tuple contains final State, compilation seconds and integration seconds;
integration timing includes actual execution, synchronization and host output.

Use ``output_callbacks(manager, step_seconds)`` to connect an OutputManager::

   from veris.execution.rollout_output import output_callbacks, run_timed

   def state_transition(state):
       return transition(state)[0]

   observe, select = output_callbacks(manager, experiment.step_duration.total_seconds())
   final, compilation_seconds, integration_seconds = run_timed(
       initial, state_transition, 100, observe=observe, select=select,
   )

These callbacks select scheduled output. Each chunk ends at the earliest
instantaneous sample, averaging-window boundary or final model time. There is
no default chunk cap; an explicit ``chunk_size`` adds a maximum number of
steps per chunk. Calendar boundaries between model times flush at the next
model time, retaining their exact original time bounds in the file.

Each mean stream carries one float64 running sum per requested field and a
sample count on the device. Unfinished sums survive chunk boundaries; no
timestep field histories are allocated for these means. Samples use the State
before each transition, so averaging windows remain half-open and obey
``sample_initial``. Only complete, nonempty windows with nonnegative starting
times are written. Partial first and final windows are discarded.
Instantaneous records use endpoint States and the initial State when requested.

Reduced fields keep their storage halos and sharding until the serial or
distributed collector removes halos and gathers physical cells. Every rank
enters collection in the same order; only the writing rank creates a file.
Custom collectors receive already averaged fields and must commute with
averaging, as halo removal and linear gathering do. A collector that applies a
nonlinear transformation per sample requires a different reduction strategy.

A scheduled run reserves a fresh manager through ``begin_reduced()`` and writes
through ``write_reduced()``. Direct ``manager.sample()`` calls and reduced runs
cannot share a manager. Reduced-mode ``close()`` closes the lazy writer; window
completion belongs to the scheduled runner. Direct sampling retains its host
running sums and calendar checks. Context-manager exit closes files on errors.

Arbitrary host observers retain the existing ``observe(sample, iteration)``
interface. They receive selected initial and subsequent samples from bounded
histories, with a default chunk cap of eight; ``chunk_size`` overrides that cap.
With no observer, including disabled OutputManager output, the runner compiles
and executes one full-length scan without observation histories and ignores
the chunk cap. Zero-step host runs perform no physics compilation or execution,
but can emit a requested initial instantaneous record. Final snapshots remain
separate host operations. Use ``veris.step`` directly for AD; the timing/output
wrapper is host-only.

See :doc:`/api/integration` for Python signatures and docstrings.
