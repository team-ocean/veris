Model data and metadata
=======================

State and diagnostics
---------------------

Create a State for a runnable experiment through
:func:`veris.initialize_experiment`;
:func:`veris.execution.initialization.initialize` creates arrays at a lower
level. State holds the model's fields, or values across the grid. Each field is
a JAX array that can change during a run. JAX treats each array as a leaf in a
PyTree, a nested collection of data. Output-only fields for exchanging results
with another model are returned in a separate Diagnostics PyTree. The statement
that they stay outside the time-stepping carry needs review: DESIGN.md describes
a rollout carry containing both State and Diagnostics. A carry is the data passed
from one step to the next. Field dimensions, units, descriptions and starting
defaults are generated from the named tables in :doc:`/reference/variables`.

.. autoclass:: veris.model.types.State

.. autoclass:: veris.model.diagnostics.Diagnostics

``veris.model.arrays.ArrayInput`` describes numerical JAX and NumPy array inputs.
``MaskInput`` also accepts NumPy boolean masks, which mark cells with true or
false values. ``veris.grid.geometry.OceanGeometry`` defines the expected grid
inputs. See :doc:`initialization` for the adapter that reads these inputs and
:doc:`experiment` for their physical dimensions.

Settings and physical constants
-------------------------------

Constructors check the values you supply and recompute quantities that depend
on them. Pass these objects as static arguments to numerical functions: their
values stay fixed while JAX compiles and runs the function. Changing a static
value can make JAX compile the function again. The tables in
:doc:`/reference/settings` and :doc:`/reference/physical-constants` describe
every field and its default. Fields computed from other inputs are not
constructor arguments.

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

The C, U, V and Z locations are cell centers for tracers, cell faces for velocity
in the x direction, cell faces for velocity in the y direction, and cell corners,
respectively. Tracers are values such as ice thickness carried across the grid.
All four dimension pairs have the same storage lengths, including halos. Halos
are extra cells around a grid piece that hold boundary or neighboring values.

.. autodata:: veris.model.variables.C_GRID

.. autodata:: veris.model.variables.U_GRID

.. autodata:: veris.model.variables.V_GRID

.. autodata:: veris.model.variables.Z_GRID
