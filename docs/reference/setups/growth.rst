Growth only
===========

The reference column starts with 1.3 m ice, 0.1 m snow, concentration 0.9,
253 K air and 80 W/m2 downward longwave radiation. It executes 150 daily Growth
steps with no dynamics or transport::

   python -m veris.setups.run_growth --backend cpu --output output/growth.nc
   python -m veris.setups.run_growth --backend gpu --output output/growth-gpu.nc

Returned Qnet/Qsw feed the next step exactly as in the original experiment.
They are not reset to initial forcing. The shared allocator requires at least
two cells per interior axis, so the single-column experiment uses a uniform
2 by 2 interior with two halos per edge; its pointwise equations are identical
to a single column. Output always removes those storage halos.
``initialize``, ``step``, ``compiled_step`` and ``step_with_diagnostics`` are also
available for interactive use and AD.

To record and plot the evolving ice thickness, request a history stream::

   python -m veris.setups.run_growth --backend cpu \
       --output output/growth.nc --netcdf output/growth-history.nc \
       --io-variables hIceMean

Then read the netCDF time series without requiring an interactive environment::

   import h5netcdf
   import matplotlib.pyplot as plt

   with h5netcdf.File("output/growth-history.nc", "r") as result:
       history = result.groups["instantaneous"]
       days = history.variables["time"][:] / 86400
       ice = history.variables["hIceMean"][:, 0, 0]
       plt.plot(days, ice)
   plt.xlabel("days")
   plt.ylabel("ice thickness / m")
   plt.show()

Reference adaptations
---------------------

The column uses the shared allocator and VARIABLES defaults for fields unused
by thermodynamics. Prescribed fields consumed by Growth retain their reference
values. Experimental constants live in the scenario registry below; universal
physical constants stay in PhysicalConstants.

Derivative validation
---------------------

The column's longwave sensitivity is checked against finite differences.
See :doc:`../automatic-differentiation` for the conventions at ice-free states
and physical branch thresholds.

Scenario controls
-----------------

.. exec::

   from veris.setups.run_growth import GROWTH_SETTINGS
   print(".. list-table:: Growth only scenario controls")
   print("   :header-rows: 1\n")
   print("   * - Setting\n     - Default\n     - Units\n     - Description")
   for name, metadata in GROWTH_SETTINGS.items():
       print(f"   * - ``{name}``\n     - ``{metadata.default!r}``")
       print(f"     - {metadata.units}\n     - {metadata.description}")
