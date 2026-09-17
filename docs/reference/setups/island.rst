Island
======

``veris.setups.island`` provides a small Cartesian sea, periodic by default, surrounding
a two-by-two-cell island. The default interior is 8 by 12 cells at 8 km spacing;
arrays include two halo cells on each side. Face masks prevent transport across
the coastline. Temperatures are in kelvin and thicknesses are grid-cell means.

.. autofunction:: veris.setups.island.initialize

.. autofunction:: veris.setups.island.step

.. autofunction:: veris.setups.island.step_with_diagnostics

The numerical State retains ice velocity and stress fields between steps.
Output-only ocean coupling stresses and fluxes are returned separately by
``step_with_diagnostics``. Its sequence is mass and area
averaging, wind forcing, ice strength, EVP dynamics, ocean stress, advection,
cleanup, ridging, growth, and boundary halo refresh. Set
``settings_overrides={"enable_cyclic_y": False}`` to close the global y edges;
see :doc:`../settings`. It restores prescribed
open-water heat forcing each step because growth returns ocean-coupling fluxes
in the same state fields.

The example initializes separate settings and physical constants and selects
serial halos. It does not initialize a distributed mesh.
The same JAX kernels can execute on a supported CPU or GPU device.

Use the public ``veris.step`` to repeat this setup kernel with scan and
checkpointing. See :doc:`../integration` for selected diagnostics and
time-varying cooling, and :doc:`/quickstart/user-guide` for a complete example.

Scenario controls
-----------------

``ISLAND_SETTINGS`` and the frozen ``IslandSettings`` class are defined
in this setup. They describe prescribed initial fields and example controls;
they do not add fields to model Configuration, PhysicalConstants or State.
This includes ``saltOcn_ref``, the prescribed ocean salinity, which can be
changed through ``scenario_overrides``.
``initialize(scenario_overrides={...})`` accepts these controls except
``islandCooling``: use ``step(..., cooling=...)`` to select that forcing.
An omitted cooling argument uses the default below.

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
