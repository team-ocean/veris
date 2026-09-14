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
The artificial example overrides timesteps and EVP iteration count.

Floating-point precision
------------------------

Choose ``dtype="float32"`` or ``dtype="float64"`` once when calling
``veris.initialization.initialize`` or the artificial initializer. Both returned
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
