Model data and metadata
=======================

State and diagnostics
---------------------

Allocate State through :func:`veris.execution.initialization.initialize` or a setup
initializer. Every State field is a dynamic JAX array leaf. Output-only
coupling fields are returned in a separate Diagnostics PyTree and stay outside
the time-stepping carry. Field dimensions, units, descriptions and allocation
defaults are generated from the registries in :doc:`/reference/variables`.

.. autoclass:: veris.model.types.State

.. autoclass:: veris.model.diagnostics.Diagnostics

``veris.model.arrays.ArrayInput`` describes numerical JAX and NumPy array inputs;
``MaskInput`` also accepts NumPy boolean masks. The geometry input schema lives
in ``veris.grid.geometry.OceanGeometry``; see :doc:`initialization` for its
adapter and :doc:`experiment` for physical input dimensions.

Settings and physical constants
-------------------------------

Constructors validate independent values and recompute derived quantities.
Bind these objects as static arguments when calling numerical kernels.
Changing static values can trigger JAX compilation. The registry tables in
:doc:`/reference/settings` and :doc:`/reference/physical-constants` document
every field and its default; derived fields are not constructor arguments.

.. autoclass:: veris.model.configuration.Configuration

.. autoclass:: veris.model.physical_constants.PhysicalConstants

Registry schemas
----------------

.. autoclass:: veris.model.types.Parameter
   :members: default, type, description, units

.. autoclass:: veris.model.variables.Variable
   :members: netcdf_attributes

.. py:data:: veris.model.configuration.SETTINGS
   :type: dict[str, veris.model.types.Parameter]

   Model settings defaults and metadata; see :doc:`/reference/settings`.

.. py:data:: veris.model.physical_constants.PHYSICALCONSTANTS
   :type: dict[str, veris.model.types.Parameter]

   Physical coefficient defaults and metadata; see
   :doc:`/reference/physical-constants`.

.. py:data:: veris.model.variables.VARIABLES
   :type: dict[str, veris.model.variables.Variable]

   Allocation and output metadata for State; see :doc:`/reference/variables`.

.. py:data:: veris.model.diagnostics.DIAGNOSTICS
   :type: dict[str, veris.model.variables.Variable]

   Metadata for output-only coupling fields; see :doc:`/reference/variables`.

Grid dimensions
---------------

The C, U, V and Z locations are tracer centers, zonal velocity faces,
meridional velocity faces and cell corners, respectively. All four dimension
pairs have the same halo-inclusive storage lengths.

.. autodata:: veris.model.variables.C_GRID

.. autodata:: veris.model.variables.U_GRID

.. autodata:: veris.model.variables.V_GRID

.. autodata:: veris.model.variables.Z_GRID
