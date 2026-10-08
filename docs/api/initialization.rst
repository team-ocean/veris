Initialization and setups
=========================

General allocation
------------------

The general allocator initializes registry defaults. Supply consistent grid
metrics, masks and forcing through ``state_overrides`` to construct a physical
experiment. For an existing example, use one of the setup initializers below.

.. autofunction:: veris.initialization.initialize

Ocean geometry adapter
----------------------

The host adapter validates externally supplied ocean geometry and allocates a
Veris State independently of an ocean-model container. See
:doc:`/reference/setups/ocean` for shapes and a usage example.

.. autoclass:: veris.setups.ocean.OceanGeometry

.. autofunction:: veris.setups.ocean.initialize_from_ocean

Island experiment
-----------------

The serial island setup combines dynamics, transport and thermodynamics.
Scenario controls are separate from model Configuration and PhysicalConstants;
their defaults are generated from ``ISLAND_SETTINGS`` in
:doc:`/reference/setups/island`. Bind ``conf`` and ``phys`` when using a setup
step as the transition for :func:`veris.step`.

.. autoclass:: veris.setups.island.IslandSettings

.. py:data:: veris.setups.island.ISLAND_SETTINGS
   :type: dict[str, veris._typing.Parameter]

   Island scenario defaults and metadata; see :doc:`/reference/setups/island`.

.. autofunction:: veris.setups.island.initialize

.. autofunction:: veris.setups.island.step

.. autofunction:: veris.setups.island.step_with_diagnostics

``compiled_step`` compiles the same complete transition as ``step``, with
``conf`` and ``phys`` static and ``cooling`` dynamic. Both support AD. Choose
and bind the callable before the integration loop to reuse compilation.

.. autofunction:: veris.setups.island.compiled_step

Dynamics experiment
-------------------

This setup advances dynamics and transport with prescribed ocean and wind
fields. Its ``nx`` and ``ny`` arguments count global physical cells; the
returned Configuration records local interior extents. The general allocator
instead accepts interior extents per partition when given a mesh. See
:doc:`/reference/setups/dynamics` and
:doc:`/reference/setups/parallel-dynamics` for the scenario registry and runs.

.. autoclass:: veris.setups.run_dyn.DynamicsSettings

.. py:data:: veris.setups.run_dyn.DYNAMICS_SETTINGS
   :type: dict[str, veris._typing.Parameter]

   Dynamics scenario defaults and metadata; see :doc:`/reference/setups/dynamics`.

.. autofunction:: veris.setups.run_dyn.initialize

.. autofunction:: veris.setups.run_dyn.step

.. autofunction:: veris.setups.run_dyn.step_with_diagnostics

``compiled_step`` compiles ``step`` with ``conf`` and ``phys`` static.

.. autofunction:: veris.setups.run_dyn.compiled_step

Growth experiment
-----------------

The serial growth column advances thermodynamics with recursive heat-flux
fields. Its ``GROWTH_SETTINGS`` defaults and command-line usage appear in
:doc:`/reference/setups/growth`.

.. autoclass:: veris.setups.run_growth.GrowthSettings

.. py:data:: veris.setups.run_growth.GROWTH_SETTINGS
   :type: dict[str, veris._typing.Parameter]

   Growth scenario defaults and metadata; see :doc:`/reference/setups/growth`.

.. autofunction:: veris.setups.run_growth.initialize

.. autofunction:: veris.setups.run_growth.step

.. autofunction:: veris.setups.run_growth.step_with_diagnostics

``compiled_step`` compiles ``step`` with ``conf`` and ``phys`` static.

.. autofunction:: veris.setups.run_growth.compiled_step
