Calendar-aware input and output
===============================

``veris.io`` writes netCDF with h5netcdf on the host. Output clocks, file handles
and averaging buffers remain outside numerical State. Each frozen ``Stream``
selects variables, a sampling interval and an averaging period; frozen
``OutputSettings`` combines streams and calendar controls. ``OutputManager``
samples concrete States after model steps and closes files through a context
manager. Existing output paths are rejected to prevent accidental overwrite.

Sampling and calendar windows
-----------------------------

Pass elapsed model time as ``datetime.timedelta``. Scheduling uses integer
microseconds, anchored to the simulation start. Integration step boundaries must
reach every sampling time exactly: a two-hour sample interval works with hourly
steps, while a ninety-minute interval does not. The manager rejects missed,
duplicate and decreasing times. Call ``sample`` at every step, including steps
without a due sample, so completed windows can flush promptly.

The ``instantaneous`` period writes individual samples. ``daily``, ``monthly``
and ``annual`` align to midnight, month starts and year starts. A positive
``timedelta`` period instead aligns to the simulation start. Windows are
half-open: a midnight sample belongs to the new day. Empty windows are skipped.

Means give every observed sample equal weight. Each mean stream retains one
float64 running sum per selected variable and a count, then writes sum/count.
Memory therefore depends on field sizes and stream count, not integration
duration. These are arithmetic sample means, not time-weighted averages.

If the simulation begins or ends inside an averaging window, that incomplete
window is discarded. This includes the initial calendar window when the model
starts after its boundary. No record or output file is created for a stream
that has no complete windows. Call ``close(final_elapsed)`` to record an explicit
endpoint; without it, context exit uses the last observed model time. Closing
at the next sampling boundary does not require a sample at that endpoint, but
closing beyond a missed sample is rejected. Exceptions during integration
discard pending averages and close the file.

Gregorian aliases ``gregorian``, ``standard`` and ``proleptic_gregorian`` use
timezone-naive standard-library ``datetime`` and are encoded as
``proleptic_gregorian``. Python's year range applies, including the upper bound
of requested averaging windows. Fixed calendars require ``FixedDate`` and use
integer calendar arithmetic instead: ``360_day`` has twelve 30-day months;
``noleap`` (alias ``365_day``) always has 365 days. Sub-day components are
supported, invalid dates are rejected, and fixed-calendar years start at one::

   from datetime import datetime, timedelta
   from veris.io import Calendar, FixedDate

   gregorian = Calendar(datetime(2000, 2, 28))
   assert gregorian.date_at(timedelta(days=1)) == datetime(2000, 2, 29)
   fixed = Calendar(FixedDate(2000, 2, 30, hour=12), "360_day")
   assert fixed.date_at(timedelta(days=1)) == FixedDate(2000, 3, 1, hour=12)

Ordinary integration
--------------------

This thermodynamic example writes daily snapshots and monthly means to separate
groups in the same file. The model timestep supplies an exactly aligned sample
interval. Initial samples are included by default::

   from datetime import timedelta
   import jax
   from veris.io import OutputManager, OutputSettings, Stream, write_snapshot
   from veris.setups.run_growth import initialize, compiled_step

   jax.config.update("jax_enable_x64", True)
   state, conf, phys = initialize()
   interval = timedelta(seconds=float(conf.deltatTherm))
   settings = OutputSettings(streams=(
       Stream("snapshots", ("hIceMean", "Area"), interval),
       Stream("monthly", ("hIceMean",), interval, "monthly"),
   ))
   with OutputManager("history.nc", settings) as output:
       output.sample(state, timedelta(0))
       for index in range(1, 31):
           state = compiled_step(state, conf, phys)
           output.sample(state, index * interval)
       output.close(30 * interval)
   write_snapshot("final.nc", state, elapsed=30 * interval, conf=conf, phys=phys)

History output always removes the two-cell serial halos. Each stream group
has its own unlimited time dimension, calendar, time units, bounds, sample
counts and coverage metadata. Field dimensions and attributes come from
:doc:`variables`, preserving staggered ``x_center``, ``x_face``, ``y_center`` and
``y_face`` names. ``cell_methods`` distinguishes point values from means.

Final snapshots and selected input
----------------------------------

``write_snapshot`` operates independently of sampling and mean buffers and
writes physical cells only, removing storage halos from every State field.
Supply ``conf`` and ``phys`` to retain configuration and physical constants as
file metadata. ``variables=("hIceMean", "Area")`` selects fields.

``read_record`` returns a ``Record`` containing named NumPy arrays and time,
calendar, coverage and snapshot metadata. Choose a group with ``stream`` and a
record with ``index`` (default: last). For example::

   from veris.io import read_record, update_state

   mean = read_record("history.nc", stream="monthly", variables=("hIceMean",))
   thickness = mean.fields["hIceMean"]
   snapshot = read_record("final.nc", variables=("hIceMean", "Area"))
   state = update_state(state, snapshot)

``update_state`` checks variable names and shapes against an initialized serial
State; the reader checks staggered dimensions. It inserts instantaneous fields
into the physical interior and preserves the existing storage halos. Refresh
those halos with the model's boundary exchange before continuing integration.
The helper rejects distributed States and mean records. Selected-field input
leaves unselected fields intact and is not a complete restart. Configuration
metadata is returned for inspection; reading does not rebuild model
initialization or output clocks.

Automatic differentiation
--------------------------

Keep differentiated integration free of output calls. If a shared host loop
retains an output manager, use ``OutputSettings(enabled=False)`` during AD.
Disabled sampling and closing return before array access, clock changes, sums,
collectives or file operations. The manager also suppresses calls inside JAX
transformations, including calls selecting only closed-over constant fields.
Standalone snapshot/input functions reject transformed use before host effects.

The guard uses the private JAX ``core.trace_state_clean`` compatibility boundary
because JAX has no public general tracing-state query. Regression coverage checks
grad, JVP and JIT; explicit disabled mode and pure numerical loops are the primary
contract when updating JAX.

Write the concrete auxiliary final State after differentiation. This example
differentiates actual thermodynamic growth with respect to initial thickness::

   from dataclasses import replace
   from datetime import timedelta
   import jax
   import jax.numpy as jnp
   from veris.io import write_snapshot
   from veris.setups.run_growth import initialize, compiled_step

   jax.config.update("jax_enable_x64", True)
   initial, conf, phys = initialize()

   def objective(scale):
       final = replace(initial, hIceMean=initial.hIceMean * scale)
       for _ in range(2):
           final = compiled_step(final, conf, phys)
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

Packed sharded States contain halos around every partition. Use
``distributed_collector(mesh)`` from ``veris.io.distributed`` to remove those
halos and gather the physical grid correctly; ordinary outer-edge slicing is
insufficient. Pass the callback as ``collector`` to both ``OutputManager`` and
``write_snapshot``. All ranks must call them with matching schedules; only rank
zero writes. Every partition's halos are removed for both histories and
snapshots. No gathering occurs during AD::

   from veris.io.distributed import distributed_collector

   collector = distributed_collector(mesh)
   with OutputManager("parallel.nc", settings, collector=collector) as output:
       output.sample(state, timedelta(0))
       # Continue the ordinary integration loop on every rank.
   write_snapshot(
       "parallel_initial.nc", state, elapsed=timedelta(0),
       conf=conf, phys=phys, collector=collector,
   )

Driver options
--------------

Maintained drivers use Click and write netCDF output. The CLI sampling default is the driver's model timestep, while
the Python ``Stream`` default is shown in the registry below.

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--netcdf PATH``
     - History output file.
   * - ``--output PATH`` (alias ``--final-netcdf``)
     - Final State snapshot; each driver supplies a default netCDF path.
   * - ``--io-variables hIceMean,Area``
     - Comma-separated State variables for history streams.
   * - ``--sample-seconds NUMBER``
     - Positive sample interval, exactly reachable by model steps.
   * - ``--average instantaneous,daily,monthly,annual``
     - Comma-separated periods, producing independent stream groups.
   * - ``--calendar gregorian|360_day|noleap``
     - Model calendar.
   * - ``--start-date YYYY-MM-DD``
     - Model origin; supports February 30 with ``360_day``.

For example::

   python -m veris.setups.run_growth --steps 30 \
       --netcdf growth_history.nc --final-netcdf growth_final.nc \
       --io-variables hIceMean,Area --average instantaneous,monthly \
       --calendar 360_day --start-date 2000-02-01

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
