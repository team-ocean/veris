Python API
==========

This reference describes the Python interface in Veris |release|.
The documentation build reads function signatures and docstrings from the code.
Start with :doc:`/introduction/get-started` to install Veris and run a first
simulation. Then use the pages below to look up individual objects and functions.
The :doc:`/reference/package-hierarchy` explains how the scientific packages are
organized and where to import specialized functions.

Use :doc:`experiment` for definitions that cannot change in place and for input
checks. Use :doc:`simulation` to control runs, output and restart from Python,
and :doc:`integration` to repeat model steps with JAX. :doc:`model-data` describes
State, which holds model arrays, Diagnostics, which holds step results, and
numerical objects that stay fixed during execution. Import specialized functions
from the modules shown in their entries.

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
