Model data and metadata
=======================

State and diagnostics
---------------------

Allocate State through :func:`veris.initialization.initialize` or a setup
initializer. Every State field is a dynamic JAX array leaf. Output-only
coupling fields are returned in a separate Diagnostics PyTree and stay outside
the time-stepping carry. Field dimensions, units, descriptions and allocation
defaults are generated from the registries in :doc:`/reference/variables`.

.. autoclass:: veris._typing.State

.. autoclass:: veris.diagnostics.Diagnostics

Settings and physical constants
-------------------------------

Constructors validate independent values and recompute derived quantities.
Bind these objects as static arguments when calling numerical kernels.
Changing static values can trigger JAX compilation. The registry tables in
:doc:`/reference/settings` and :doc:`/reference/physical-constants` document
every field and its default; derived fields are not constructor arguments.

.. autoclass:: veris.configuration.Configuration

.. autoclass:: veris.physical_constants.PhysicalConstants

Registry schemas
----------------

.. autoclass:: veris._typing.Parameter
   :members: default, type, description, units

.. autoclass:: veris.variables.Variable
   :members: netcdf_attributes

.. py:data:: veris.configuration.SETTINGS
   :type: dict[str, veris._typing.Parameter]

   Model settings defaults and metadata; see :doc:`/reference/settings`.

.. py:data:: veris.physical_constants.PHYSICALCONSTANTS
   :type: dict[str, veris._typing.Parameter]

   Physical coefficient defaults and metadata; see
   :doc:`/reference/physical-constants`.

.. py:data:: veris.variables.VARIABLES
   :type: dict[str, veris.variables.Variable]

   Allocation and output metadata for State; see :doc:`/reference/variables`.

.. py:data:: veris.diagnostics.DIAGNOSTICS
   :type: dict[str, veris.variables.Variable]

   Metadata for output-only coupling fields; see :doc:`/reference/variables`.

Grid dimensions
---------------

The C, U, V and Z locations are tracer centers, zonal velocity faces,
meridional velocity faces and cell corners, respectively. All four dimension
pairs have the same halo-inclusive storage lengths.

.. autodata:: veris.variables.C_GRID

.. autodata:: veris.variables.U_GRID

.. autodata:: veris.variables.V_GRID

.. autodata:: veris.variables.Z_GRID
