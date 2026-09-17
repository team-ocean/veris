Scan rollouts and checkpointing
===============================

``veris.step`` advances a fixed number of model timesteps using
`jax.lax.scan <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>`_.
A setup supplies the single-step physics; the shared driver supplies time
iteration. Bind Configuration and PhysicalConstants with ``functools.partial``.
The default result is the final State, without an explicit State history::

   from functools import partial
   import jax
   jax.config.update("jax_enable_x64", True)

   from veris import step
   from veris.setups import artificial

   initial, settings, constants = artificial.initialize()
   advance = partial(artificial.step, conf=settings, phys=constants, cooling=100.0)
   final = step(initial, advance, 3)

``steps`` is a static nonnegative integer. The transition must be pure and JAX
traceable: carry structure, array shapes and dtypes must stay fixed. Reuse bound
callables across rollouts to reuse compiled scans. Settings and physical
constants keep their existing static treatment; State and forcing arrays are
differentiable. Python mutation, NumPy conversions and file output belong
outside the transition and observer.

Setup choices
-------------

The same driver accepts the maintained setup kernels:

* ``artificial.step`` composes dynamics, transport and growth. Its ``cooling``
  argument restores prescribed heat forcing on every timestep.
* ``run_dyn.step`` performs dynamics and transport with prescribed ocean and
  wind fields. ``run_parallel`` uses this same kernel with a mesh.
* ``run_growth.step`` performs thermodynamics only. Returned ``Qnet`` and
  ``Qsw`` remain recursive inputs to the next step, as in the reference column.
* ``ocean.initialize_from_ocean`` supplies geometry and initial fields, rather
  than a separate time-stepping scheme. Bind a physics kernel appropriate to
  those supplied fields, for example ``artificial.step`` for coupled evolution.

Both ``artificial.step`` and ``run_dyn.step`` use the core
``veris.dynamics.dynamics_transport(state, settings, constants)`` stage. It
advances momentum, computes ocean stress before transport, then applies
advection, cleanup and ridging. It returns ``(state, stress_u, stress_v)`` with
halo-inclusive arrays. Callers supply atmospheric forcing, optional
thermodynamic growth, the sharding context and the final State/Diagnostics halo
refresh. Under sharding, this local stage runs inside the caller's
``jax.shard_map``.

For example, a growth rollout is::

   from veris.setups import run_growth

   initial, settings, constants = run_growth.initialize()
   advance = partial(run_growth.step, conf=settings, phys=constants)
   final = step(initial, advance, 3)

Selected observations and diagnostics
-------------------------------------

A pure ``observe(state)`` function selects an array PyTree after each updated
State. With an observer, the result is ``(final_state, history)``. Each history
leaf has a leading time axis of length ``steps``; the initial State is excluded::

   final, history = step(
       initial, advance, 3,
       observe=lambda state: {"ice": state.hIceMean[2:-2, 2:-2]},
   )

For output-only coupling fields, use ``has_aux=True`` with a transition returning
``(state, diagnostics)`` and an observer accepting both arguments::

   advance = partial(run_growth.step_with_diagnostics, conf=settings, phys=constants)
   final, history = step(
       initial, advance, 3, has_aux=True,
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
``steps``. The transition then receives ``advance(state, input_slice)``. For
example, differentiate a thickness objective with respect to each prescribed
cooling value::

   import jax
   import jax.numpy as jnp

   initial, settings, constants = artificial.initialize()
   advance = partial(artificial.step, conf=settings, phys=constants)
   cooling = jnp.asarray([80.0, 100.0, 120.0], dtype=initial.hIceMean.dtype)

   def objective(forcing):
       final = step(initial, advance, 3, inputs=forcing, checkpoint=True)
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
transformations. The existing setup kernel owns halo exchange and local stencil
calculations; the scan driver does not repartition the grid::

   import jax
   from veris.setups import run_dyn

   mesh = jax.make_mesh((1, jax.device_count()), ("x", "y"))
   with jax.set_mesh(mesh):
       initial, settings, constants = run_dyn.initialize(
           8, 8 * jax.device_count(), mesh=mesh,
       )
       advance = partial(run_dyn.step, conf=settings, phys=constants)
       final = step(initial, advance, 3)

These initializer dimensions count global physical cells. Each partition keeps
its own halos. Selected history arrays gain an unpartitioned leading time axis;
select one time slice before using the existing spatial output collector.
Sharded artificial cooling accepts spatially uniform scalar forcing, including
one replicated scalar per time slice. Its convenience initializer remains
serial; use the general allocator with consistent packed fields for sharded
coupled experiments, as described in :doc:`/quickstart/user-guide`.

Host output and timing
----------------------

The dynamics, growth and parallel command-line runners use
``veris.integration_output.run_timed``. It lowers and compiles each required
scan shape with ``jax.jit(...).lower(...).compile()`` before executing the
trajectory. Compilation can trace pure callables, but executes no model
transitions, device callbacks, sample collection or file writes. The returned
tuple contains final State, compilation seconds and integration seconds;
integration timing includes actual execution, synchronization and host output.

Use ``output_callbacks(manager, step_seconds)`` to connect an OutputManager::

   from veris.integration_output import output_callbacks, run_timed

   observe, select = output_callbacks(manager, settings.deltatDyn)
   final, compilation_seconds, integration_seconds = run_timed(
       initial, advance, 100, observe=observe, select=select,
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

.. autofunction:: veris.step
