Python API
==========

This reference documents the Python interface on the ``jax-only`` branch.
Signatures and docstrings are read from the implementation at documentation
build time. Start with :doc:`/quickstart/user-guide` for a working simulation,
then use the pages below to look up individual objects and functions.
The :doc:`/reference/package-hierarchy` describes the scientific packages and
the owning modules for specialized imports.

Model initialization returns ``(state, settings, constants)``. The frozen
:class:`~veris.model.types.State` contains the array fields used by the numerical
kernels; :class:`~veris.model.configuration.Configuration` and
:class:`~veris.model.physical_constants.PhysicalConstants` hold static controls and
coefficients. Use ``dataclasses.replace`` to update these immutable objects.

The rollout driver :func:`veris.step` and host definition classes ``Grid``,
``Execution``, ``Forcing`` and ``Experiment`` are exported directly from ``veris``.
``OceanGeometry`` and ``initialize_experiment`` provide runnable initialization;
``advance`` performs a single physics timestep through that same public import.
``Simulation`` owns the host lifecycle and returns ``RunResult``; see
:doc:`simulation` for successive runs, forcing position and optional histories.
See :doc:`experiment` for definition and input
contracts. Import other functions and
classes from the modules shown in their entries.
For example::

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

Arrays use ``(x, y)`` order with two halo cells on each boundary. A serial
``nx`` by ``ny`` physical grid has storage shape ``(nx + 4, ny + 4)``. Under
sharding, each partition has its own halos; see :doc:`initialization` for the
different dimension conventions of the general and dynamics initializers.
Enable JAX x64 before requesting ``float64`` initialization.

Pure kernels and :func:`veris.step` can be used in JAX transformations. Keep
host allocation, timing and file I/O outside differentiated rollouts. See
:doc:`/reference/automatic-differentiation` for derivative conventions at
physical thresholds and zero norms.

.. toctree::
   :maxdepth: 1

   model-data
   experiment
   simulation
   initialization
   integration
   physics
   parallel
   io
