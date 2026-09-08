.. _variables:

State fields
============

``veris.variables.variables`` is a dictionary of field names with ``None``
placeholders. It does not allocate arrays or carry restart, dimension, or unit
metadata. The artificial example allocates every field as a two-dimensional
JAX array and adds ``forc_salt_surface`` for the growth salt-flux output.

For an interior of ``nx`` by ``ny`` cells, example arrays have shape
``(nx + 4, ny + 4)``. The interior is ``field[2:-2, 2:-2]``. Axis zero is zonal;
axis one is meridional. Fields use staggered cell-center and face locations.

Key fields
----------

* ``hIceMean``, ``hSnowMean``: grid-cell mean thicknesses in metres.
* ``Area``: ice-covered fraction; ``AreaW`` and ``AreaS`` are face averages.
* ``uIce``, ``vIce``: face-centered ice velocities in m/s.
* ``sigma1``, ``sigma2``, ``sigma12``: stress fields retained between steps.
* ``TSurf``, ``theta``, ``ATemp``: surface, ocean, and air temperatures in kelvin.
* ``iceMask``, ``iceMaskU``, ``iceMaskV``: ocean cell and face masks.
* ``Qnet``, ``Qsw``: heat fluxes in W/m², positive upward. Growth replaces the
  supplied forcing values with ocean-coupling output fluxes.

Field comments in ``veris/variables.py`` and kernel docstrings describe the
remaining fields. State updates return a new named tuple in the artificial
example; use ``state._replace(field_name=array)`` for prescribed changes.

Registry names
--------------

.. exec::

   from veris.variables import variables
   for name in variables:
       print(f"* ``{name}``")
