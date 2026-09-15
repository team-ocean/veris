Artificial island
=================

``veris.setups.artificial`` provides a small periodic Cartesian sea surrounding
a two-by-two-cell island. The default interior is 8 by 12 cells at 8 km spacing;
arrays include two halo cells on each side. Face masks prevent transport across
the coastline. Temperatures are in kelvin and thicknesses are grid-cell means.

.. autofunction:: veris.setups.artificial.initialize

.. autofunction:: veris.setups.artificial.step

.. autofunction:: veris.setups.artificial.step_with_diagnostics

The numerical State retains ice velocity and stress fields between steps.
Output-only ocean coupling stresses and fluxes are returned separately by
``step_with_diagnostics``. Its sequence is mass and area
averaging, wind forcing, ice strength, EVP dynamics, ocean stress, advection,
cleanup, ridging, growth, and periodic halo refresh. It restores prescribed
open-water heat forcing each step because growth returns ocean-coupling fluxes
in the same state fields.

The example initializes separate settings and physical constants and selects
serial halos. It does not initialize a distributed mesh.
The same JAX kernels can execute on a supported CPU or GPU device.

Scenario controls
-----------------

``ARTIFICIAL_SETTINGS`` and the frozen ``ArtificialSettings`` class are defined
in this setup. They describe prescribed initial fields and example controls;
they do not add fields to model Configuration, PhysicalConstants or State.
This includes ``saltOcn_ref``, the prescribed ocean salinity, which can be
changed through ``scenario_overrides``.
``initialize(scenario_overrides={...})`` accepts these controls except
``artificialCooling``: use ``step(..., cooling=...)`` to select that forcing.
An omitted cooling argument uses the default below.

.. exec::

   from veris.setups.artificial import ARTIFICIAL_SETTINGS
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Setting")
   print("     - Default")
   print("     - Type")
   print("     - Description")
   for name, metadata in ARTIFICIAL_SETTINGS.items():
       print(f"   * - ``{name}``")
       print(f"     - ``{metadata.default!r}``")
       print(f"     - ``{metadata.type.__name__}``")
       print(f"     - {metadata.description}")
