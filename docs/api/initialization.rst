Initialization and presets
==========================

Common experiment initialization
--------------------------------

``initialize_experiment`` creates model arrays for a serial run on a single
device or for the parts of a device mesh that the current process can access.
A mesh groups devices that share the grid. Registered providers supply starting
field values for each grid
region. Veris checks inputs, applies overrides, computes quantities from other
inputs and handles the outer grid boundaries. The maintained presets specify
their archived startup rules so all 70 initial fields remain reproducible.
The island, dynamics and growth presets all use this initializer. Set the grouped
controls through ``Experiment`` before creating arrays::

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

The general allocator creates arrays using defaults from the model's named
parameter tables. To define a physical experiment, supply consistent grid
distances and areas, masks that mark wet cells, and prescribed external inputs
through ``state_overrides``. For an existing example, construct its builtin
Experiment and use ``initialize_experiment``.

.. autofunction:: veris.execution.initialization.initialize

Ocean geometry adapter
----------------------

The adapter checks ocean grid information supplied from outside Veris. It runs
in Python and creates a Veris State without requiring an ocean-model container.
See :doc:`/reference/setups/ocean` for array shapes and a usage example.
Masks must lie in [0, 1], grid spacing and area must be positive, and prescribed
ice amounts, ice-covered fractions and temperatures must satisfy the Experiment
physical bounds. Before creating device arrays, Python checks also reject numbers
or their reciprocals that the selected numerical precision cannot represent.

Supply ``latitude`` (ny centers in degrees), ``latitude_edges`` (ny+1 edges)
and ``radius`` (positive metres) together to select spherical geometry. These
coordinate vectors exclude halos in both direct and Experiment inputs. Omit all
three for the existing planar contract. Zonal spacing vectors describe
equatorial distances; supplied tracer and face areas retain their own quadrature.
The adapter derives the geometry-owned static ``spherical_geometry`` setting.
It must agree with the selected geometry. Spherical inputs default to closed
latitude boundaries; explicitly cyclic latitude is rejected because strictly
monotone, nonsingular latitude coordinates cannot form a periodic ring.

Spherical normal strain has a finite-volume trace using those stored tracer
areas. Tensor divergence is its negative area-weighted adjoint, including the
maintained slip stencil. Corner dual areas use the actual corner lengths and
wet quadrants, so a straight wall owns half a dual cell. Immutable face metrics
retain the true northern face through closed-wall halo refresh. Singular polar
edges, partial coordinates and invalid radius/coordinate shapes fail on the host.

.. autoclass:: veris.grid.geometry.OceanGeometry

.. autofunction:: veris.grid.ocean.initialize_from_ocean

Island experiment
-----------------

The serial island setup combines dynamics, which computes ice motion, transport,
which carries ice and snow across the grid, and thermodynamics, which computes
freezing and melting. Scenario controls are separate from model Configuration
and PhysicalConstants. Their defaults are generated from ``ISLAND_SETTINGS`` in
:doc:`/reference/setups/island`. Use :func:`veris.advance` for one combined update
without Python side effects and :func:`veris.step` to repeat a fixed number of
updates.

.. autoclass:: veris.setups.island.IslandSettings

.. py:data:: veris.setups.island.ISLAND_SETTINGS
   :type: dict[str, veris.model.types.Parameter]

   Island scenario defaults and metadata; see :doc:`/reference/setups/island`.

Dynamics experiment
-------------------

This setup computes ice motion and transport using supplied ocean and wind
fields. Set the full grid dimensions with ``Grid``. The Configuration returned
by ``initialize_experiment`` records the interior size of each local grid piece.
When given a mesh, the general allocator instead accepts interior sizes per
partition, or grid piece. See
:doc:`/reference/setups/dynamics` and
:doc:`/reference/setups/parallel-dynamics` for the scenario registry and runs.

.. autoclass:: veris.setups.run_dyn.DynamicsSettings

.. py:data:: veris.setups.run_dyn.DYNAMICS_SETTINGS
   :type: dict[str, veris.model.types.Parameter]

   Dynamics scenario defaults and metadata; see :doc:`/reference/setups/dynamics`.

Growth experiment
-----------------

The serial growth column computes freezing and melting. Its heat-flux fields
describe heat exchanged with the ice and feed results from one step into the
next. Its ``GROWTH_SETTINGS`` defaults and command-line usage appear in
:doc:`/reference/setups/growth`.

.. autoclass:: veris.setups.run_growth.GrowthSettings

.. py:data:: veris.setups.run_growth.GROWTH_SETTINGS
   :type: dict[str, veris.model.types.Parameter]

   Growth scenario defaults and metadata; see :doc:`/reference/setups/growth`.

Select ``physics="thermodynamics"`` with :func:`veris.advance` to update the
initialized column, or use its builtin Experiment with Simulation.
