Python API
==========

This reference documents the Python interface on the ``jax-only`` branch.
Signatures and docstrings are read from the implementation at documentation
build time. Start with :doc:`/introduction/get-started` for installation and a first simulation,
then use the pages below to look up individual objects and functions.
The :doc:`/reference/package-hierarchy` describes the scientific packages and
the owning modules for specialized imports.

Use :doc:`experiment` for immutable definitions and input validation,
:doc:`simulation` for host execution, output and restart, and
:doc:`integration` for pure JAX rollouts. :doc:`model-data` describes State,
Diagnostics and static numerical objects. Import specialized functions from
the owning modules shown in their entries.

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
