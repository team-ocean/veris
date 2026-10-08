Python API
==========

This reference documents the Python interface on the ``jax-only`` branch.
Signatures and docstrings are read from the implementation at documentation
build time. Start with :doc:`/quickstart/user-guide` for a working simulation,
then use the pages below to look up individual objects and functions.

Model initialization returns ``(state, settings, constants)``. The frozen
:class:`~veris._typing.State` contains the array fields used by the numerical
kernels; :class:`~veris.configuration.Configuration` and
:class:`~veris.physical_constants.PhysicalConstants` hold static controls and
coefficients. Use ``dataclasses.replace`` to update these immutable objects.

Only the rollout driver :func:`veris.step` is exported directly from ``veris``.
Import other functions and classes from the modules shown in their entries.
For example::

   from functools import partial
   import jax

   jax.config.update("jax_enable_x64", True)

   from veris import step
   from veris.setups import island

   initial, settings, constants = island.initialize(nx=8, ny=8)
   advance = partial(island.step, conf=settings, phys=constants, cooling=100.0)
   final = step(initial, advance, 3)

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
   initialization
   integration
   physics
   parallel
   io
