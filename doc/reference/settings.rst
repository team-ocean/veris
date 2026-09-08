Model settings
==============

``veris.settings.settings`` is a dictionary of plain default values. Compiled
kernels accept a hashable settings object as their ``sett`` argument; the
artificial example supplies an immutable named tuple with attribute access.

For example::

   from veris.setup.artificial import initialize
   state, settings = initialize()
   settings = settings._replace(nEVPsteps=20)

The example overrides several registry defaults, including timesteps, halo mode,
and EVP iteration count. A settings change may trigger JAX recompilation.
Numerical meanings and units are recorded beside values in ``veris/settings.py``.

Registry defaults
-----------------

.. exec::

   from veris.settings import settings
   print(".. list-table::")
   print("   :header-rows: 1")
   print("")
   print("   * - Setting")
   print("     - Default")
   for name, value in settings.items():
       print(f"   * - ``{name}``")
       print(f"     - ``{value!r}``")
