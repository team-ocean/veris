Model settings
==============

``veris.configuration.SETTINGS`` defines defaults, scalar types and descriptions
for the frozen ``Configuration`` dataclass. Numerical controls and execution choices
are separate from :doc:`physical-constants`. Grid extents, boundary-condition
switches, timesteps, solver iterations and convergence safeguards stay in
Configuration. Physical thresholds, regularization scales and forcing reference
heights are fields of PhysicalConstants.

Use immutable updates; dependent reciprocals are recomputed::

   from dataclasses import replace
   from veris.configuration import Configuration
   settings = replace(Configuration(), deltatDyn=600, nEVPsteps=20)
   assert settings.recip_deltatDyn == 1 / 600

Configuration objects are static JIT arguments; changing a value may trigger compilation.
The island example overrides timesteps and EVP iteration count.

Horizontal boundaries
---------------------

The x direction is always periodic. ``enable_cyclic_y=True`` (the default)
also wraps the global y edges. Set it to ``False`` at initialization to use
closed, impermeable southern and northern walls::

   from veris.setups.island import initialize
   state, settings, constants = initialize(
       settings_overrides={"enable_cyclic_y": False}
   )

Closed walls retain every interior tracer row. Exterior masks are dry, normal
ice velocities and meridional transport vanish at both walls, and ``noSlip``
selects the existing tangential coastline treatment. This is a solid-wall
condition, not an open boundary or a polar-fold grid.

Serial and sharded execution use the same conditions. Only the global y edge
partitions apply walls; internal partitions continue exchanging their halos.
The static switch adds no State leaves and supports JVP and VJP differentiation.
Generic scalar halos extend the nearest interior value so temperatures and
metrics remain valid. Masks, ice amounts, velocities and fluxes receive their
corresponding wall conditions.
Wall shear stress is handled separately: no-slip retains the calculated
traction, including the northern physical face stored in a halo column;
free-slip sets both wall shear stresses to zero.

Select the boundary topology at initialization. Reinitialize the geometry and
masks when changing it: switching a closed State back to periodic does not
restore the southern face masks that were closed during allocation.

After manually replacing State fields or loading selected physical fields,
refresh the whole State
before integration using ``veris.fill_overlap.fill_state_overlap(state, settings)``.
For sharded arrays, call it inside the active ``jax.set_mesh(mesh)`` context.
The maintained coupled drivers refresh State and diagnostic halos after each
step. Low-level custom drivers should use this field-aware helper instead of
applying the generic scalar halo fill to every field.
Selected-field snapshots are not complete restarts (see :doc:`io`). In
particular, trimming halos omits the northern no-slip shear-stress value;
halo refresh preserves that independent traction but cannot reconstruct it.

Floating-point precision
------------------------

Choose ``dtype="float32"`` or ``dtype="float64"`` once when calling
``veris.initialization.initialize`` or the island initializer. Both returned
static objects retain this policy. Their floating coefficients, derived values
and lookup tables use the selected NumPy scalar type; all state and work arrays
use the matching JAX dtype. Integers and Boolean switches retain their types.
``float64`` requires JAX x64 enabled before initialization. Veris rejects silent
truncation and does not change global JAX configuration. Immutable replacement
preserves precision. Array replacements supplied by callers must use the existing
array dtype. NetCDF storage uses the actual array dtype, as shown in :doc:`variables`.

Precision is defined directly by the ``Parameter`` entry at
``veris.configuration.SETTINGS["dtype"]``. Configuration objects declare their own keyword-only
``dtype`` field using that same default; there is no precision base class.

Registry defaults
-----------------

.. exec::

   from veris.configuration import SETTINGS
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
