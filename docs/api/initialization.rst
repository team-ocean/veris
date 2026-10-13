Initialization and presets
==========================

Common experiment initialization
--------------------------------

``initialize_experiment`` allocates both serial and addressable mesh regions
through the registered provider contract. New providers supply regional field
replacements; the framework validates inputs, applies overrides, computes
derived quantities and handles global boundaries. The maintained presets declare
their archived startup policy so all 70 initial fields remain reproducible.
The island, dynamics and growth presets all use this initializer. Define grouped
controls through ``Experiment`` before allocating arrays::

   import jax
   from veris import Experiment, Grid, initialize_experiment

   jax.config.update("jax_enable_x64", True)
   experiment = Experiment.builtin(
       "island", grid=Grid(8, 12),
       settings={"enable_cyclic_y": False},
       scenario={"islandWindSpeed": 5.0},
       constants={"rhoIce": 900.0},
   )
   state, settings, constants = initialize_experiment(experiment)

.. autofunction:: veris.initialize_experiment

.. autoclass:: veris.setups.presets.SetupPreset

.. autofunction:: veris.setups.presets.register_preset

General allocation
------------------

The general allocator initializes registry defaults. Supply consistent grid
metrics, masks and forcing through ``state_overrides`` to construct a physical
experiment. For an existing example, construct its builtin Experiment and use
``initialize_experiment``.

.. autofunction:: veris.execution.initialization.initialize

Ocean geometry adapter
----------------------

The host adapter validates externally supplied ocean geometry and allocates a
Veris State independently of an ocean-model container. See
:doc:`/reference/setups/ocean` for shapes and a usage example.
Masks must lie in [0, 1], spacing and area must be positive, and prescribed
ice amounts, concentration and temperatures obey the Experiment physical bounds.
Host validation also rejects values or reciprocals outside the selected
precision before allocating device arrays.

.. autoclass:: veris.grid.geometry.OceanGeometry

.. autofunction:: veris.grid.ocean.initialize_from_ocean

Island experiment
-----------------

The serial island setup combines dynamics, transport and thermodynamics.
Scenario controls are separate from model Configuration and PhysicalConstants;
their defaults are generated from ``ISLAND_SETTINGS`` in
:doc:`/reference/setups/island`. Use :func:`veris.advance` for the pure coupled
transition and :func:`veris.step` for a fixed-length rollout.

.. autoclass:: veris.setups.island.IslandSettings

.. py:data:: veris.setups.island.ISLAND_SETTINGS
   :type: dict[str, veris.model.types.Parameter]

   Island scenario defaults and metadata; see :doc:`/reference/setups/island`.

Dynamics experiment
-------------------

This setup advances dynamics and transport with prescribed ocean and wind
fields. Define global physical dimensions with ``Grid``; the Configuration
returned by ``initialize_experiment`` records local interior extents. The general
allocator instead accepts interior extents per partition when given a mesh. See
:doc:`/reference/setups/dynamics` and
:doc:`/reference/setups/parallel-dynamics` for the scenario registry and runs.

.. autoclass:: veris.setups.run_dyn.DynamicsSettings

.. py:data:: veris.setups.run_dyn.DYNAMICS_SETTINGS
   :type: dict[str, veris.model.types.Parameter]

   Dynamics scenario defaults and metadata; see :doc:`/reference/setups/dynamics`.

Growth experiment
-----------------

The serial growth column advances thermodynamics with recursive heat-flux
fields. Its ``GROWTH_SETTINGS`` defaults and command-line usage appear in
:doc:`/reference/setups/growth`.

.. autoclass:: veris.setups.run_growth.GrowthSettings

.. py:data:: veris.setups.run_growth.GROWTH_SETTINGS
   :type: dict[str, veris.model.types.Parameter]

   Growth scenario defaults and metadata; see :doc:`/reference/setups/growth`.

Select ``physics="thermodynamics"`` with :func:`veris.advance` to evolve the
initialized column, or use its builtin Experiment with Simulation.
