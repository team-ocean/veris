Island
======

``veris.setups.island`` provides a small Cartesian sea on a rectangular x/y
grid, surrounding a two-by-two-cell island. It is periodic by default: each
outer edge connects to its opposite edge. The default interior is 8 by 12
cells at 8 km spacing. Arrays include two halo cells on each side, which hold
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
