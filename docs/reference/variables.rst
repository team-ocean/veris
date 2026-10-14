.. _variables:

State fields
============

``veris.model.variables.VARIABLES`` describes the calculation fields created by
``veris.execution.initialization.initialize``. Every entry has a default,
units, long name, description and dimensions that identify its grid location.
The grid is staggered: some values sit at cell centers and others at cell
faces. State is a frozen dataclass, whose fields cannot be reassigned, and a
JAX PyTree, a collection of arrays that JAX can process together. Update arrays
by creating a new State with ``dataclasses.replace``.

For an interior of ``nx`` by ``ny`` cells, horizontal arrays have shape
``(nx + 4, ny + 4)``. The interior is ``field[2:-2, 2:-2]``. Axis zero is zonal
(east-west); axis one is meridional (north-south). Dimensions identify Cartesian
grid locations: ``x_center``, ``y_center``, ``x_face`` and ``y_face``. Face and
center arrays have equal storage sizes. Both include two halo cells at each
boundary, which hold boundary or neighboring data. When arrays are split
across devices, each partition keeps its own halos; see
:doc:`/api/experiment`. Boundary conditions are described in :doc:`settings`.

State holds the geometry fields and prescribed forcing used by calculations.
Output-only diagnostics and configuration do not add arrays to numerical
State. Arrays for ice-thickness categories exist only as intermediate values
in the thermodynamics calculations, which describe freezing and melting.

NetCDF metadata
---------------

The metadata can be used directly with h5netcdf::

   from veris.model.variables import VARIABLES
   metadata = VARIABLES["uIce"]
   variable = output.create_variable(
       "uIce", metadata.dimensions, dtype=state.uIce.dtype
   )
   variable.attrs.update(metadata.netcdf_attributes())
   variable[:] = state.uIce

Here ``output`` is an open ``h5netcdf.File`` with the horizontal dimensions
already defined. The example writes the halos along with the interior. When
exporting only the interior, trim the arrays and adjust the dimension lengths
together.

Registry defaults
-----------------

.. exec::

   from veris.model.variables import VARIABLES
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

``advance`` returns these fields in a separate frozen dataclass. These
diagnostics describe the exchange with an ocean model and do not add arrays to
the calculation State. When ice salinity is nonzero, the two terms in the
legacy combined salt forcing use different freshwater normalizations (scaling
factors). Its units are recorded as ``unknown`` to avoid misleading units. The
move to this metadata keeps the existing equation.

.. exec::

   from veris.model.diagnostics import DIAGNOSTICS
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
