Island
======

``veris.setups.island`` provides a Cartesian sea on an 80 by 120 x/y grid,
surrounding a 20 by 20 cell island, with 800 m spacing. It is periodic by
default: each outer edge connects to its opposite edge. Arrays include two
halo cells on each side, which hold
boundary or neighboring data. Face masks identify blocked cell faces to
prevent transport across the coastline. Temperatures use kelvin, the absolute
temperature scale. Thicknesses are grid-cell means, averaged over each cell's
whole area.

See :func:`veris.initialize_experiment` in the
:doc:`Python API </api/initialization>` for creating preset arrays, and
:func:`veris.advance` for the complete pure timestep. A pure timestep returns
results without changing outside resources.

The numerical State, the collection of model arrays, keeps ice velocity and
stress fields between steps. Stress is force per area. ``advance`` returns
output-only ocean coupling stresses and fluxes (transfers between ice and
ocean) separately. Its sequence is mass and area averaging, wind forcing, ice
strength, EVP dynamics, ocean stress, advection, cleanup, ridging, growth, and
boundary halo refresh. EVP means elastic-viscous-plastic, the ice-motion
solution method. Advection moves values between cells; ridging piles up ice;
growth calculates freezing and melting. Set
``settings={"enable_cyclic_y": False}`` on the Experiment to close the global y
edges; see :doc:`../settings`. The timestep restores prescribed open-water
heat forcing each step because growth returns ocean-coupling fluxes in the
same state fields.

Use ``Experiment.builtin("island", grid=Grid(nx, ny), ...)`` with
``Simulation.initialize`` for shared initialization, prescribed heat, forcing
and time counted from the start of the simulation, and output. The public run
interface supports a single device and explicit x/y device meshes with global
physical dimensions. The caller prepares JAX resources; see
:doc:`/quickstart/user-guide` and :doc:`/api/simulation`.

Coupled Experiments combine ice motion with freezing and melting. They require
equal dynamics and thermodynamic durations so their single model clock is
unambiguous.

Use the public ``veris.step`` to repeat the coupled transition with a scan
(a timestep loop) and checkpointing (recalculating intermediate values to save
memory during reverse differentiation). See :doc:`../integration` for selected
diagnostics and time-varying cooling, and :doc:`/quickstart/user-guide` for a
complete example.

Island example
--------------

The sole ``example`` profile uses 80 by 120 cells with 800 m spacing.
It preserves the original 64 by 96 km
domain and the 16 by 16 km island (256 square km), now resolved by 20 by 20
land cells. The island still spans x = 24--40 km and y = 40--56 km. This is a
Cartesian grid; its axes measure distance rather than latitude and longitude.

Both physics timesteps are 60 s, with 400 adaptive EVP iterations per step.
The ordinary island defaults and the named ``example`` use this same grid and
numerical configuration. There is no separate low-resolution or ``refined``
profile. ``veris run island`` selects this example and defaults to 300 steps.

All 300 steps retain finite model fields, zero ice on land and blocked coastal
velocities, and periodic halos. The maximum sum of directional Courant numbers
is 0.01174. At the same five-hour duration, doubling EVP iterations to 800
changes concentration by at most 0.00171 and halving the timestep to 30 s
changes it by at most 0.000567. These are stability and sensitivity checks for
this example, rather than a general convergence result.

.. figure:: /figures/setups/island-concentration-300-steps.svg
   :alt: Final ice concentration around the unchanged 16-by-16-km central island, with sparse vermilion wind arrows and a stationary prescribed ocean.
   :width: 720px
   :align: center

   Ice concentration after 300 coupled steps of 60 s (5 h), using float64 on
   CPU, an 80 by 120 physical grid and 400 adaptive EVP iterations per step.
   Colours show the ice-covered fraction of each ocean cell; grey marks the
   island. Vermilion arrows show atmospheric wind, with a 5 m/s reference
   arrow. The prescribed ocean velocity is zero everywhere, so no ocean
   streamlines are drawn. Storage halos are excluded. The basin and island
   retain their original physical areas. This figure reuses the saved run.

Download the :download:`SVG figure </figures/setups/island-concentration-300-steps.svg>`.
It uses outlined lettering, vector wind arrows and an embedded raster map,
so its typography does not depend on fonts installed on the reader's device.

Reproduce the plotted final state with::

   python -m veris.cli run island --steps 300 --backend cpu \
       --output-snapshot output/island.nc \
       --io-variables Area,maskInC,uOcean,vOcean,uWind,vWind

Plot the snapshot's ``Area[0, :, :]`` from the ``snapshot`` group, masking cells
where ``maskInC[0, :, :]`` is zero. Transpose the masked array to put x on the
horizontal axis. Physical cell edges are spaced by 800 m from zero in both
directions, giving a 64 by 96 km domain. Keep the concentration colour scale
fixed from zero to one.

After running all three documented setups into ``output/``, regenerate their
SVG figures from the repository root with::

   python docs/examples/plot_setup_results.py output

Scenario controls
-----------------

This setup defines ``ISLAND_SETTINGS`` and the frozen ``IslandSettings`` class,
whose fields cannot be reassigned. They describe prescribed initial fields
and example controls. They do not add fields to model Configuration,
PhysicalConstants or State. These controls include ``saltOcn_ref``, the
prescribed ocean salinity, which you can change through
``Experiment.builtin("island", scenario={...})``. Use
``initial_conditions={...}`` to replace physical fields directly. Prescribe
heat through the Experiment forcing definition or the ``forcing`` argument of
``advance``. The island built-in setup supplies the heat default below.

.. exec::

   from veris.setups.island import ISLAND_SETTINGS
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Setting")
   print("     - Default")
   print("     - Type")
   print("     - Description")
   for name, metadata in ISLAND_SETTINGS.items():
       print(f"   * - ``{name}``")
       print(f"     - ``{metadata.default!r}``")
       print(f"     - ``{metadata.type.__name__}``")
       print(f"     - {metadata.description}")
