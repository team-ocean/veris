Growth only
===========

The reference column starts with 1.3 m ice, 0.1 m snow, concentration 0.9,
253 K air and 80 W/m2 downward longwave radiation. It executes 150 daily Growth
steps with no dynamics or transport::

   python -m veris.setups.run_growth --backend cpu --output output/growth.nc
   python -m veris.setups.run_growth --backend gpu --output output/growth-gpu.nc

The CLI uses ``Experiment.builtin("growth", ...)`` and ``Simulation``. Python
experiments use the same interface for initial conditions, constant/series
forcing, output and complete serial restart. The builtin defaults to Grid(2, 2);
custom global dimensions and explicit meshes are also supported. See
:doc:`/quickstart/user-guide` and :doc:`/api/simulation`.

Returned Qnet/Qsw feed the next step exactly as in the original experiment.
They are not reset to initial forcing. The shared allocator requires at least
two cells per interior axis, so the single-column experiment uses a uniform
2 by 2 interior with two halos per edge; its pointwise equations are identical
to a single column. Output always removes those storage halos.
The preset ``initialize`` factory constructs the initial column. Use public
``advance(..., physics="thermodynamics")`` and ``step`` for pure evolution and AD.

The default experiment reproduces the saved ``jax_halo_exchange`` notebook:
ice decreases from 1.3 m and disappears at the end of day 123. The notebook
plots the values before each step (days 0--149); scheduled instantaneous output
includes the initial value and completed steps (days 0--150).

To record and plot the evolving ice thickness, request a history stream::

   python -m veris.setups.run_growth --backend cpu \
       --output output/growth-history-final.nc --netcdf output/growth-history.nc \
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

The column zero-initializes unprescribed fields exactly as
``initialize_growth.py`` does. Prescribed fields and physical constants retain
the reference values. Experimental inputs live in the scenario registry below;
physical constants stay in PhysicalConstants.

The notebook's original Growth equations multiply three already grid-cell mean
fluxes by concentration a second time: surface snow melt, penetrating shortwave
and ocean energy loss. The reference preset explicitly selects these equations
with ``useLegacyGrowthAreaWeights=True``. They reproduce the historical experiment
and its partial-cover budget errors. The general ``Configuration`` default is
``False``, retaining the corrected concentration weights and conservation tests.
Using corrected equations with the same column inputs moves complete melt to
day 140::

   from veris import Experiment, Simulation

   definition = Experiment.builtin(
       "growth", steps=150,
       settings={"useLegacyGrowthAreaWeights": False},
   )
   with Simulation.initialize(definition) as simulation:
       result = simulation.run()

All eleven thermodynamic outputs over 150 days are checked against a frozen
capture of the original ``jax-only`` kernels at
``eb690aa8dc25ed7e6bb3766d658f382eca234b7c`` and standalone reference
``jax_halo_exchange`` commit ``2571998f8feb6a5943b214361c122ca78b44c3a8``.

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
