Model setups
============

The public Experiment and Simulation interfaces run every built-in setup with
the same initialization, forcing, timekeeping, input and output. Forcing means
prescribed inputs such as wind or ocean heat. See :doc:`/quickstart/user-guide`
and :doc:`/quickstart/custom-ocean` for complete public workflows. Setup
factories create the preset fields, and command-line drivers follow the same
run process. The central :doc:`/cli` lists, copies and runs editable preset
definitions.

The island setup combines dynamics (ice motion) and growth (freezing and
melting) with prescribed ocean masks and fields. Masks identify ocean and land
cells. It requires no geographic datasets or external ocean model. The
separate dynamics, growth and parallel dynamics cases use the same
initialization and physics calculations. Each page describes its controls and
usage. The three separate cases derive from ``veris_minimum_working_example``
branch ``jax_halo_exchange`` (reference commit
``2571998f8feb6a5943b214361c122ca78b44c3a8``).

For installation and a first run, see :doc:`/introduction/get-started`.
Use ``--help`` for command-line options. Importing a setup does not run a simulation.

.. toctree::
   :maxdepth: 1

   setups/island
   setups/dynamics
   setups/growth
   setups/parallel-dynamics

New experiments use the shared :doc:`/quickstart/setup-blueprint`, either with
prescribed ocean geometry or a registered factory that creates fields for each
grid region. Python and CLI launch the same public runner. Exports of the
user's inputs and of the values after defaults are applied keep the setup
identity and the order in which overrides take priority.
