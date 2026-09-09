Model settings
==============

``veris.configuration.SETTINGS`` defines defaults, scalar types and descriptions
for the frozen ``Settings`` dataclass. Numerical controls and execution choices
are separate from :doc:`physical-constants`.

Use immutable updates; dependent reciprocals are recomputed::

   from dataclasses import replace
   from veris.configuration import Settings
   settings = replace(Settings(), deltatDyn=600, nEVPsteps=20)
   assert settings.recip_deltatDyn == 1 / 600

Settings are static JIT arguments; changing a value may trigger compilation.
The artificial example overrides timesteps and EVP iteration count.

Registry defaults
-----------------

.. exec::

   from veris.configuration import SETTINGS
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Setting")
   print("     - Default")
   print("     - Type")
   print("     - Description")
   for name, metadata in SETTINGS.items():
       print(f"   * - ``{name}``")
       print(f"     - ``{metadata.default!r}``")
       print(f"     - ``{metadata.type.__name__}``")
       print(f"     - {metadata.description}")
