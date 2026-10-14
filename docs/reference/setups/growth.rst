Growth only
===========

The reference column represents ice and snow above ocean water. It starts
with 1.3 m ice, 0.1 m snow, concentration 0.9 (the fraction of each cell covered
by ice), 253 K air and 80 W/m2 downward longwave radiation (heat radiated from the
atmosphere). It executes 150 daily Growth steps, calculating freezing and
melting without ice motion or transport between cells::

   python -m veris.setups.run_growth --backend cpu --output-snapshot output/growth.nc
   python -m veris.setups.run_growth --backend gpu --output-snapshot output/growth-gpu.nc

The CLI uses ``Experiment.builtin("growth", ...)`` and ``Simulation``. Python
experiments use the same interface for initial conditions, forcing (prescribed
inputs that are constant or vary with time), output and complete serial
restart. The built-in setup defaults to Grid(2, 2). You can also choose global
dimensions and explicit device meshes. See :doc:`/quickstart/user-guide` and
:doc:`/api/simulation`.

Returned Qnet/Qsw feed the next step exactly as in the original experiment.
They are not reset to initial forcing. The shared allocator requires at least
two cells per interior axis, so the single-column experiment uses a uniform
2 by 2 interior with two halos per edge. Halos are extra cells for boundary or
neighboring data. The equations at each grid point are identical to those of
a single column. Output always removes those storage halos.
``initialize_experiment(Experiment.builtin("growth", grid=Grid(2, 2)))``
creates the initial column. Use public
``advance(..., physics="thermodynamics")`` and ``step`` to advance it with pure
functions and automatic differentiation (AD). Pure functions do not change
outside resources; AD calculates how results change with inputs.

The setup uses the ``jax_halo_exchange`` notebook's initial conditions and
daily steps with the shared Growth routine. With the default inputs, ice
decreases from 1.3 m and disappears at the end of day 140. The notebook plots
values before each step (days 0--149). Scheduled instantaneous output includes
the initial value and values after completed steps (days 0--150).

To record and plot the evolving ice thickness, request a history stream::

   python -m veris.setups.run_growth --backend cpu \
       --output-snapshot output/growth-history-final.nc --output-history output/growth-history.nc \
       --io-variables hIceMean

Then read the netCDF time series without needing an interactive environment::

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

The column starts fields that were not prescribed at zero, exactly as
``initialize_growth.py`` does. Prescribed fields and physical constants keep
the reference values. Experiment-specific inputs live in the scenario registry
below; physical constants stay in PhysicalConstants.

Grid dimensions and run duration belong to ``Grid`` and ``Experiment.steps``.
``GrowthSettings`` and ``GROWTH_SETTINGS`` contain only the physical column
scenario. The field provider and run profiles keep the 2 by 2 grid and 150-step
defaults. Scenario and physical overrides use the same grouped definition as
other presets::

   import jax
   from veris import Experiment, Grid, initialize_experiment

   jax.config.update("jax_enable_x64", True)
   experiment = Experiment.builtin(
       "growth", grid=Grid(2, 2), scenario={"hIceMean": 1.3},
       constants={"rhoIce": 900.0},
   )
   state, settings, constants = initialize_experiment(experiment)

Tests check initial fields and shared settings/constants against the
standalone reference ``jax_halo_exchange`` commit
``2571998f8feb6a5943b214361c122ca78b44c3a8``. The column evolves with the same
thermodynamic calculations as other Veris experiments. These include heat and
snow budgets that account for cells covered only partly by ice.

Derivative validation
---------------------

Tests check the column's sensitivity to longwave radiation with finite
differences, which compare results after small input changes. See
:doc:`../automatic-differentiation` for derivative choices when ice is absent
and at thresholds where the model selects a different physical branch.

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
