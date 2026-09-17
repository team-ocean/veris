Model setups
============

The island setup combines dynamics and growth with prescribed ocean masks
and fields. It requires no geographic datasets or external ocean model.
The isolated dynamics, growth and parallel dynamics cases reuse the same
initialization and physics kernels. Each page describes its controls and usage.
The three isolated cases derive from ``veris_minimum_working_example`` branch
``jax_halo_exchange`` (reference commit
``2571998f8feb6a5943b214361c122ca78b44c3a8``).

Activate the installed environment from the project root::

   source .venv-latest/bin/activate

The runners select float64 at runtime and write netCDF output. Use ``--help``
for command-line options. Importing a setup does not run a simulation.

.. toctree::
   :maxdepth: 1

   setups/island
   setups/ocean
   setups/dynamics
   setups/growth
   setups/parallel-dynamics
