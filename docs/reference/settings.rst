Model settings
==============

``veris.model.configuration.SETTINGS`` defines defaults, scalar types (types of
individual values) and descriptions for the frozen ``Configuration``
dataclass. Its fields cannot be reassigned. Numerical controls and execution
choices are separate from :doc:`physical-constants`. Configuration holds grid
sizes, switches for behavior at grid edges, timestep lengths, solver iteration
counts and safeguards for convergence (when repeated calculations settle on a
solution). PhysicalConstants holds physical thresholds, scales that keep
equations well behaved, and reference heights for atmospheric forcing.

Create an updated object instead of changing fields in place. A reciprocal is
one divided by a value. Reciprocals calculated from changed values are recalculated::

   from dataclasses import replace
   from veris.model.configuration import Configuration
   settings = replace(Configuration(), deltatDyn=600, nEVPsteps=20)
   assert settings.recip_deltatDyn == 1 / 600

JAX treats Configuration objects as fixed inputs during JIT compilation, which
turns Python calculations into executable code. Changing a value may require
compilation again. The island example changes timestep lengths and the
iteration count of the EVP (elastic-viscous-plastic) ice-motion solver.

Horizontal boundaries
---------------------

The x direction is always periodic: crossing one edge returns to the opposite
edge. ``enable_cyclic_y=True`` (the default) also joins the global y edges this
way. Set it to ``False`` at initialization to use closed southern and northern
walls that nothing can pass through::

   import jax
   from veris import Experiment, Grid, initialize_experiment

   jax.config.update("jax_enable_x64", True)
   experiment = Experiment.builtin(
       "island", grid=Grid(8, 12), settings={"enable_cyclic_y": False},
   )
   state, settings, constants = initialize_experiment(experiment)

Closed walls keep every interior row of tracers, such as ice amount and
temperature. Exterior masks mark land. Ice velocity perpendicular to the wall
and north-south transport are zero at both walls. ``noSlip`` selects the
existing coastline treatment for motion along the wall. These are solid walls;
they do not form an open boundary or a grid folded across a pole.

Single-device execution and execution split across devices use the same
conditions. Only partitions at the global y edges apply walls. Internal
partitions still exchange their halos, the extra cells that hold boundary or
neighboring data. The fixed switch adds no arrays to State and supports JVP
(following a chosen input change forward) and VJP (carrying output sensitivities
back to inputs). Generic
scalar halos copy the nearest interior value so temperatures and grid metrics
remain valid. Masks, ice amounts, velocities and fluxes use their own wall
conditions. Wall shear stress, the force per area along the wall, is handled
separately. No-slip keeps the calculated traction, including the northern
physical face stored in a halo column. Free-slip sets both wall shear stresses
to zero.

Choose how the grid edges connect at initialization. Reinitialize the geometry
and masks when changing that choice. Switching a closed State back to periodic
does not restore the southern face masks that were closed during allocation.

After manually replacing State fields or loading selected physical fields,
refresh the whole State before stepping with
``veris.grid.halos.fill_state_overlap(state, settings)``. For arrays split
across devices, call it inside the active ``jax.set_mesh(mesh)`` context.
The maintained coupled drivers refresh State and diagnostic halos after each
step. Custom low-level drivers should use this helper, which applies each
field's boundary rules, instead of filling every field as a generic scalar.
Selected-field snapshots are not complete restarts (see :doc:`io`). Removing
halos omits the northern no-slip shear-stress value. Halo refresh preserves
that independent traction but cannot reconstruct a missing value.

Floating-point precision
------------------------

Choose ``settings={"dtype": "float32"}`` or ``settings={"dtype": "float64"}``
when defining an Experiment, then call ``initialize_experiment``. These types
store floating-point numbers at different precisions. The general allocator
also accepts ``dtype`` directly. Both returned static objects keep this choice.
Their floating coefficients, calculated values and lookup tables use the
selected NumPy scalar type. All state and work arrays use the matching JAX
dtype. Integers (whole numbers) and Boolean switches (true or false) keep their
types. ``float64`` requires
JAX x64 enabled before initialization. Veris rejects silent loss of precision
and does not change global JAX configuration. Creating an updated object keeps
its precision. Arrays supplied as replacements must use the existing array
dtype. NetCDF storage uses the actual array dtype, as shown in :doc:`variables`.

The ``Parameter`` entry at ``veris.model.configuration.SETTINGS["dtype"]``
defines precision directly. Configuration objects declare their own
keyword-only ``dtype`` field with that default. They do not inherit precision
from a base class.

Configuration keeps ``explicitDrag`` because version-one complete restart files
store the full set of settings fields. The current solvers do not read this
field, so changing it leaves the numerical calculation unchanged.

Registry defaults
-----------------

.. exec::

   from veris.model.configuration import SETTINGS
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Setting")
   print("     - Default")
   print("     - Unit")
   print("     - Description")
   for name, metadata in SETTINGS.items():
       print(f"   * - ``{name}``")
       print(f"     - ``{metadata.default!r}``")
       units = "``-``" if metadata.units in ("1", "-", "") else metadata.units
       print(f"     - {units}")
       print(f"     - {metadata.description}")
