Island
======

``veris.setups.island`` provides a small Cartesian sea, periodic by default, surrounding
a two-by-two-cell island. The default interior is 8 by 12 cells at 8 km spacing;
arrays include two halo cells on each side. Face masks prevent transport across
the coastline. Temperatures are in kelvin and thicknesses are grid-cell means.

See :func:`veris.initialize_experiment` in the
:doc:`Python API </api/initialization>` for preset allocation, and
:func:`veris.advance` for its complete pure timestep.

The numerical State retains ice velocity and stress fields between steps.
Output-only ocean coupling stresses and fluxes are returned separately by
``advance``. Its sequence is mass and area
averaging, wind forcing, ice strength, EVP dynamics, ocean stress, advection,
cleanup, ridging, growth, and boundary halo refresh. Set
``settings={"enable_cyclic_y": False}`` on the Experiment to close the global y
edges; see :doc:`../settings`. It restores prescribed
open-water heat forcing each step because growth returns ocean-coupling fluxes
in the same state fields.

Use ``Experiment.builtin("island", grid=Grid(nx, ny), ...)`` with
``Simulation.initialize`` for shared initialization, prescribed heat, absolute
forcing/time and output. The public lifecycle supports serial and explicit
x/y meshes with global physical dimensions. The caller prepares JAX resources;
see :doc:`/quickstart/user-guide` and :doc:`/api/simulation`.

Coupled Experiments require equal dynamics and thermodynamic durations so
their single model clock is unambiguous.

Use the public ``veris.step`` to repeat the coupled transition with scan and
checkpointing. See :doc:`../integration` for selected diagnostics and
time-varying cooling, and :doc:`/quickstart/user-guide` for a complete example.

Scenario controls
-----------------

``ISLAND_SETTINGS`` and the frozen ``IslandSettings`` class are defined
in this setup. They describe prescribed initial fields and example controls;
they do not add fields to model Configuration, PhysicalConstants or State.
This includes ``saltOcn_ref``, the prescribed ocean salinity, which can be
changed through ``Experiment.builtin("island", scenario={...})``.
Use ``initial_conditions={...}`` for direct physical field replacements.
Prescribe heat through the Experiment forcing definition or the ``forcing``
argument of ``advance``; the island builtin supplies the heat default below.

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
