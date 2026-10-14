Calendar-aware input and output
===============================

``veris.io`` uses h5netcdf to write netCDF files, which store arrays and their
metadata. It runs on the host, the CPU running the Python program. Output
clocks, open files and averaging buffers stay outside numerical State, the
collection of model arrays. Each frozen ``Stream`` selects variables, a
sampling interval and an averaging period. Frozen ``OutputSettings`` combines
streams and calendar controls. Frozen objects cannot have their fields
reassigned. ``OutputManager`` samples States with available array values after
model steps and uses a context manager to close files. It rejects existing
output paths to prevent accidental overwrite.

Floating-point precision
------------------------

Model initialization selects ``float32`` or ``float64`` through ``dtype``;
the default is ``float64``. These floating-point number formats store numbers
with limited precision. Float64 stores more digits than float32. Output uses
the following precision policy:

.. list-table::
   :header-rows: 1

   * - Part
     - Precision
   * - Model fields, geometry and floating coefficients
     - Selected model dtype: ``float32`` or ``float64``
   * - Instantaneous output and final snapshot fields
     - Same dtype as the model
   * - Time-average accumulators and saved means
     - Always ``float64``
   * - NetCDF time coordinates and bounds
     - Always ``float64``

Float64 accumulation reduces rounding error when summing many samples,
especially when positive and negative contributions cancel. It cannot recover
precision already lost during float32 model calculations. Averaging buffers
stay separate from numerical State, so they do not change model precision or
evolution. For float32 runs, mean buffers and saved mean fields use twice the
bytes per value compared with float32 fields.

.. _output-sampling:

Sampling and calendar windows
-----------------------------

This section describes direct ``OutputManager.sample`` calls. For managed
``Simulation`` output and the timed runner, see :ref:`scheduled-output-integration`.

Pass elapsed model time as ``datetime.timedelta``. The schedule counts integer
microseconds from the simulation start. Model steps must land exactly on
every sampling time: a two-hour sample interval works with hourly steps, while
a ninety-minute interval does not. The manager rejects missed times, duplicate
times and times that go backward. Call ``sample`` at every step, including
steps when no sample is due, so it can write completed averaging windows
promptly.

The ``instantaneous`` period writes individual samples. ``daily``, ``monthly``
and ``annual`` align to midnight, month starts and year starts. A positive
``timedelta`` period instead aligns to the simulation start. Each averaging
window includes its start and excludes its end: a midnight sample belongs to
the new day. Empty windows are skipped.

Means give every observed sample equal weight. Each mean stream keeps one
float64 running sum per selected variable and a count, then writes sum/count.
Memory therefore depends on field sizes and stream count rather than run
duration. These arithmetic sample means do not weight samples by the time
between them.

If the simulation begins or ends inside an averaging window, that incomplete
window is discarded. This includes the first calendar window when the model
starts after its boundary. A stream with no complete windows creates no record
or output file. Call ``close(final_elapsed)`` to give an explicit end time.
Otherwise, leaving the context uses the last observed model time. Closing at
the next sampling boundary does not require a sample at that endpoint.
Closing beyond a missed sample is rejected. If an exception interrupts the
run, pending averages are discarded and the file closes.

The Gregorian calendar aliases ``gregorian``, ``standard`` and
``proleptic_gregorian`` use standard-library ``datetime`` values without time
zones. Files encode them as ``proleptic_gregorian``. Python's year range applies
to all dates, including the end of requested averaging windows. Fixed
calendars require ``FixedDate`` and use integer calendar arithmetic instead:
``360_day`` has twelve 30-day months; ``noleap`` (alias ``365_day``) always has
365 days. Dates can include times within a day. Invalid dates are rejected,
and fixed-calendar years start at one::

   from datetime import datetime, timedelta
   from veris.io import Calendar, FixedDate

   gregorian = Calendar(datetime(2000, 2, 28))
   assert gregorian.date_at(timedelta(days=1)) == datetime(2000, 2, 29)
   fixed = Calendar(FixedDate(2000, 2, 30, hour=12), "360_day")
   assert fixed.date_at(timedelta(days=1)) == FixedDate(2000, 3, 1, hour=12)

Direct sampling
---------------

This thermodynamic (freezing and melting) example writes daily snapshots and
monthly means to separate groups in the same file. The sampling interval
matches the model timestep exactly. Initial samples are included by default::

   from datetime import timedelta
   import jax
   from veris.io import OutputManager, OutputSettings, Stream, write_snapshot
   from veris import Experiment, Grid, advance, initialize_experiment

   jax.config.update("jax_enable_x64", True)
   state, conf, phys = initialize_experiment(
       Experiment.builtin("growth", grid=Grid(2, 2))
   )
   interval = timedelta(seconds=float(conf.deltatTherm))
   settings = OutputSettings(streams=(
       Stream("snapshots", ("hIceMean", "Area"), interval),
       Stream("monthly", ("hIceMean",), interval, "monthly"),
   ))
   with OutputManager("history.nc", settings) as output:
       output.sample(state, timedelta(0))
       for index in range(1, 31):
           state, _ = advance(state, conf, phys, physics="thermodynamics")
           output.sample(state, index * interval)
       output.close(30 * interval)
   write_snapshot("final.nc", state, elapsed=30 * interval, conf=conf, phys=phys)

History output always removes the two-cell serial halos, the extra boundary
cells around the physical grid. Each stream group has its own time dimension
that can grow, calendar, time units, bounds, sample counts and coverage
metadata. Field dimensions and attributes come from :doc:`variables`. They
keep the staggered grid names ``x_center``, ``x_face``, ``y_center`` and
``y_face``, which identify cell centers and faces. ``cell_methods``
distinguishes values at a single time from means.

.. _scheduled-output-integration:

Scheduled integration
----------------------

For ordinary experiments, attach ``OutputSettings`` to an ``Experiment`` and
let ``Simulation`` sample it; see :doc:`/quickstart/user-guide`. This uses
``run_timed`` and ``output_callbacks`` internally. Custom host drivers can
connect those helpers directly as shown in :doc:`integration`.

These callbacks select scheduled output. A chunk is a group of model steps.
Each chunk ends at the earliest instantaneous sample, averaging-window
boundary or final model time. There is no default limit on chunk length. An
explicit ``chunk_size`` sets a maximum number of steps per chunk. If a calendar
boundary falls between model times, its window is written at the next model
time, with the exact original time bounds in the file.

Each mean stream keeps one float64 running sum per requested field and a
sample count on the device. Unfinished sums pass from one chunk to the next.
These means do not need saved histories of fields at every timestep. Sampling
follows ``sample_initial`` and the complete-window rule above.

Fields prepared for output, including accumulated means, keep their storage
halos and device partitions until the serial or distributed collector removes
halos and gathers physical cells. Every rank, or participating process, enters
collection in the same order. Only the writing rank creates a file. Custom
collectors receive already averaged fields. Collecting before averaging must
give the same result as collecting after averaging, as it does for halo
removal and linear gathering. A collector that applies a nonlinear operation
to each sample needs a different way to accumulate results.

A scheduled run reserves a fresh manager through ``begin_reduced()`` and writes
through ``write_reduced()``. Direct ``manager.sample()`` calls and reduced runs
cannot share a manager. Reduced-mode ``close()`` closes the writer, which opens
a file only when needed. The scheduled runner completes the averaging windows.
Direct sampling keeps its running sums and calendar checks on the host.
Leaving the context manager closes files on errors.

Instantaneous records use States after completed steps and, when requested,
the initial State once. Mean samples use fields before each step, so the rules
above for window endpoints and complete windows also apply to managed output.
Successive ``Simulation.run`` calls keep unfinished sums/counts and the
original output schedule. Closing discards an unfinished window. Output and
pure observation histories can use the same model run. The run interface
saves definitions/settings/constants with defaults and overrides applied as
file attributes.

Final snapshots and selected input
----------------------------------

``write_snapshot`` runs independently of sampling and mean buffers. It writes
only physical cells, removing storage halos from every State field. Supply
``conf`` and ``phys`` to save configuration and physical constants as file
metadata. ``variables=("hIceMean", "Area")`` selects fields.

``read_record`` returns a ``Record`` with named NumPy arrays and metadata for
time, calendar, coverage and snapshots. Choose a group with ``stream`` and a
record with ``index`` (default: last). For example::

   from veris.io import read_record, update_state

   mean = read_record("history.nc", stream="monthly", variables=("hIceMean",))
   thickness = mean.fields["hIceMean"]
   snapshot = read_record("final.nc", variables=("hIceMean", "Area"))
   state = update_state(state, snapshot)

``update_state`` checks variable names and shapes against an initialized
serial State. The reader checks dimensions for cell-center and cell-face
locations. The helper inserts instantaneous fields into the physical interior
and keeps the existing storage halos. Refresh those halos with the model's
boundary exchange before stepping again. The helper rejects distributed
States and mean records. Loading selected fields keeps unselected fields
intact and is not a complete restart. Configuration metadata is returned for
inspection. Reading does not initialize the model again or rebuild output
clocks.

Automatic differentiation
--------------------------

Automatic differentiation (AD) calculates how results change when inputs
change. Keep output calls outside differentiated model steps. If a shared host
loop keeps an output manager, use ``OutputSettings(enabled=False)`` during AD.
Disabled sampling and closing return before accessing arrays, changing clocks,
calculating sums, communicating across processes or working with files. The
manager also suppresses calls inside JAX transformations, including calls that
select only constant fields captured by the function. Standalone snapshot/input
functions reject use inside transformations before doing anything on the host.

The safeguard uses the private JAX ``core.trace_state_clean`` interface because
JAX has no public general way to ask whether it is tracing a calculation.
Regression tests check grad (gradients), JVP (forward directional derivatives)
and JIT (compilation). Explicit disabled mode and pure numerical loops are the
main supported approach when updating JAX.

Write the final State returned as an auxiliary result after differentiation,
when its array values are available. This example calculates how thermodynamic
growth changes with initial thickness::

   from dataclasses import replace
   from datetime import timedelta
   import jax
   import jax.numpy as jnp
   from veris.io import write_snapshot
   from veris import Experiment, Grid, advance, initialize_experiment

   jax.config.update("jax_enable_x64", True)
   initial, conf, phys = initialize_experiment(
       Experiment.builtin("growth", grid=Grid(2, 2))
   )

   def objective(scale):
       final = replace(initial, hIceMean=initial.hIceMean * scale)
       for _ in range(2):
           final, _ = advance(final, conf, phys, physics="thermodynamics")
       return jnp.mean(final.hIceMean[2:-2, 2:-2]), final

   (mean_thickness, final), sensitivity = jax.value_and_grad(
       objective, has_aux=True
   )(1.0)
   write_snapshot(
       "after_ad.nc", final,
       elapsed=timedelta(seconds=2 * float(conf.deltatTherm)),
       conf=conf, phys=phys,
   )

Distributed output
------------------

States packed from device partitions contain halos around every partition.
Use ``distributed_collector(mesh)`` from ``veris.io.distributed`` to remove
those halos and gather the physical grid correctly. Removing only the outer
array edges is insufficient. Pass the callback as ``collector`` to both
``OutputManager`` and ``write_snapshot``. All ranks must call them with matching
schedules; only rank zero writes. Every partition's halos are removed for both
histories and snapshots. No gathering occurs during AD::

   from veris.io.distributed import distributed_collector

   collector = distributed_collector(mesh)
   with OutputManager("parallel.nc", settings, collector=collector) as output:
       output.sample(state, timedelta(0))
       # Continue the ordinary integration loop on every rank.
   write_snapshot(
       "parallel_initial.nc", state, elapsed=timedelta(0),
       conf=conf, phys=phys, collector=collector,
   )

Command-line output
--------------------

See :doc:`/cli` for final snapshots, sampled histories and module-driver output
options. Output stream defaults are generated below.

Registry defaults
-----------------

The following tables are generated from the registries used to initialize
``OutputSettings`` and ``Stream``. These host controls are separate from model
Configuration and physical constants.

.. exec::

   from veris.io import OUTPUT_SETTINGS, STREAM_SETTINGS
   for title, registry in (("OutputSettings", OUTPUT_SETTINGS), ("Stream", STREAM_SETTINGS)):
       print(f".. list-table:: {title}")
       print("   :header-rows: 1")
       print("")
       print("   * - Option")
       print("     - Default")
       print("     - Type")
       print("     - Description")
       for name, metadata in registry.items():
           print(f"   * - ``{name}``")
           print(f"     - ``{metadata.default!r}``")
           print(f"     - ``{metadata.type.__name__}``")
           print(f"     - {metadata.description}")
       print("")
