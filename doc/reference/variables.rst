.. _variables:

State fields
============

``veris.variables.VARIABLES`` describes the calculation fields allocated by
``veris.initialization.initialize``. Every entry has a default, dtype, units,
long name, description and staggered dimensions. State is a frozen dataclass
registered as a JAX PyTree. Update arrays with ``dataclasses.replace``.

For an interior of ``nx`` by ``ny`` cells, horizontal arrays have shape
``(nx + 4, ny + 4)``. The interior is ``field[2:-2, 2:-2]``. Axis zero is zonal;
axis one is meridional. Dimensions describe Cartesian grid locations:
``x_center``, ``y_center``, ``x_face`` and ``y_face``. Face and center storage
extents are equal, including two periodic halo cells at each boundary.

Geometry and prescribed forcing used by calculations belong to State.
Output-only diagnostics and configuration do not add numerical State leaves.
Ice-thickness category arrays are local intermediates in thermodynamics.

NetCDF metadata
---------------

The metadata can be used directly with h5netcdf::

   from veris.variables import VARIABLES
   metadata = VARIABLES["uIce"]
   variable = output.create_variable(
       "uIce", metadata.dimensions, dtype=metadata.dtype
   )
   variable.attrs.update(metadata.netcdf_attributes())
   variable[:] = state.uIce

Here ``output`` is an open ``h5netcdf.File`` with the horizontal dimensions
already defined. The example writes halo-inclusive storage; trim arrays and
adjust dimension lengths together when exporting only the interior.

Registry defaults
-----------------

.. exec::

   from veris.variables import VARIABLES
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Variable")
   print("     - Dimensions")
   print("     - Units")
   print("     - Default")
   print("     - Description")
   for name, metadata in VARIABLES.items():
       print(f"   * - ``{name}``")
       print(f"     - {', '.join(metadata.dimensions)}")
       print(f"     - {metadata.units}")
       print(f"     - ``{metadata.default!r}``")
       print(f"     - {metadata.description}")

Coupling diagnostics
--------------------

``step_with_diagnostics`` returns these fields in a separate frozen dataclass.
They do not add leaves to the calculation State. The legacy combined salt
forcing has differing freshwater normalizations in its two terms when ice
salinity is nonzero; its units are recorded as ``unknown`` rather than assigning
misleading units. This metadata migration preserves that existing equation.

.. exec::

   from veris.diagnostics import DIAGNOSTICS
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Variable")
   print("     - Dimensions")
   print("     - Units")
   print("     - Default")
   print("     - Description")
   for name, metadata in DIAGNOSTICS.items():
       print(f"   * - ``{name}``")
       print(f"     - {', '.join(metadata.dimensions)}")
       print(f"     - {metadata.units}")
       print(f"     - ``{metadata.default!r}``")
       print(f"     - {metadata.description}")
