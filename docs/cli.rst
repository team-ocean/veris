Command line tools
==================

After :doc:`installation <introduction/get-started>`, the ``veris`` command is
available from any directory::

   veris --help
   veris --version

You can also use ``python -m veris.cli`` with the same arguments. The central
interface uses Click, a Python tool for building command line interfaces. It
follows the steps for finding, copying and running setups used by the
`Vercor command line tools <https://vercor.readthedocs.io/en/latest/cli.html>`_.
It runs Veris's existing :doc:`Experiment and Simulation lifecycle
<quickstart/user-guide>`.

All commands
------------

.. exec::

   from click.testing import CliRunner
   from veris.cli import cli
   print("::\n")
   result = CliRunner().invoke(cli, ["--help"], prog_name="veris")
   assert result.exit_code == 0, result.output
   print("\n".join("   " + line for line in result.output.splitlines()))

Show preconfigured setups
-------------------------

List the available editable templates::

   veris show-setups

The builtin templates are ``island`` (ice motion, transport, freezing and melting
with supplied ocean masks and fields), ``dynamics`` (a basin with fixed wind) and
``growth`` (freezing and melting in a column with supplied conditions). Masks
identify which grid cells contain ocean. Fields hold values, such as temperature,
across the grid. See :doc:`reference/setup-gallery` for the physical assumptions
and calculation methods. All use the maintained code to prepare the preset's
starting conditions. Copying a template does not run a model or allocate its arrays.

Copy a selected setup
---------------------

Copy a template and its relative assets to an editable directory::

   veris copy-setup island --to ./my-island

The destination must be new. Without ``--to``, the destination is ``./SETUP``
in the current directory. Parent directories are created as needed. Existing
directories, files and symlinks (links to other filesystem paths) are rejected.
This protects existing user edits.

Additional template roots can be supplied through ``VERIS_SETUP_DIR``. Each
root contains one folder per setup, with a ``setup.py`` file and optional
data or helper files::

   extra-setups/
     my-basin/
       setup.py
       masks.nc

On Linux/macOS, separate several roots with ``:``; Windows uses ``;``::

   export VERIS_SETUP_DIR=/path/to/extra-setups:/path/to/other-setups
   veris show-setups
   veris copy-setup my-basin --to ./experiment
   veris inspect my-basin
   veris run my-basin --backend cpu --steps 3 --output-snapshot basin.nc

Names are sorted and must be unique across builtin and external roots.
Missing roots and duplicate names produce errors. Discovery inspects only
these directories and does not import their Python files.

To create your own setup outside the Veris repository, copy a builtin under
your chosen name into an external root, then edit its ``setup.py``::

   veris copy-setup island --to /path/to/extra-setups/my-sea-ice
   export VERIS_SETUP_DIR=/path/to/extra-setups
   veris show-setups
   veris inspect my-sea-ice
   veris run my-sea-ice --backend cpu --steps 3 --output-snapshot my-sea-ice.nc

The folder name is the command line interface (CLI) setup name. Avoid names ending in ``.py`` or
``.json``, which identify files. Set ``VERIS_SETUP_DIR`` in each shell that
should discover it. The runner uses an existing file before looking up a setup
name. It rejects directories and symlinks whose targets are missing. Explicit paths (including
``./`` prefixes) and missing ``.py``/``.json`` files never fall back to a setup
name. Runs selected by name use the same setup-building function, overrides and
output behavior as runs selected by file. Overrides replace values in the
definition. Builtin names can also be run directly, for example ``veris run growth``.

Run a setup
-----------

Run a copied Python setup, optionally overriding execution controls::

   veris run ./my-island/setup.py --backend cpu --steps 3 --output-snapshot island.nc

Output paths are relative to the current working directory. A final snapshot
defaults to ``output.nc`` and contains all native State fields at the final
absolute model time, measured from the model's origin. Halos are extra cells
around the grid edges used for neighboring values; output removes them. Use fresh destinations for repeated
runs. ``--io-variables hIceMean,Area`` selects snapshot fields.

.. exec::

   from click.testing import CliRunner
   from veris.cli import cli
   print("::\n")
   result = CliRunner().invoke(cli, ["run", "--help"], prog_name="veris")
   assert result.exit_code == 0, result.output
   print("\n".join("   " + line for line in result.output.splitlines()))

Python setup contract
~~~~~~~~~~~~~~~~~~~~~

A Python setup defines ``build_experiment``, a function that takes no arguments
and returns a validated ``Experiment``. This function is called a factory
because it builds the experiment definition. The :doc:`/quickstart/setup-blueprint`
provides a complete factory and matching Python/CLI runs.

Edit the definition to supply numerical settings, physical constants, initial
conditions, forcing, geometry or output schedules. Forcing means supplied inputs
that drive the model, such as wind or heat. Geometry describes the grid's layout
and cell sizes. The factory builds definitions on the host, the Python process
that controls the calculation. The runner allocates arrays on the computing
device and runs the model. Python setup files execute trusted user code. They
can import helper modules in the same folder. Locate input files relative to
``Path(__file__).parent``. The runner
keeps the caller's working directory and does not execute an
``if __name__ == "__main__"`` block. Existing standalone setup drivers remain
available through their documented ``python -m veris.setups...`` commands.

JSON and overrides
~~~~~~~~~~~~~~~~~~

The same runner accepts :doc:`versioned JSON definitions <api/experiment>`::

   veris run experiment.json --backend cpu --output-snapshot final.nc

Explicit CLI flags override steps, dtype, backend and mesh; omitted flags
preserve the definition's values. See :doc:`api/experiment` for the order in
which overrides apply. Float64, a 64-bit floating-point number format, is the
model default. ``--float-type`` updates the number format for both numerical
settings and physical constants. The runner enables JAX x64 (64-bit calculations)
before allocating arrays. This includes the float64 sums used by sampled output
even when the physics uses float32, a 32-bit number format.
Help, version, discovery and copy commands do not change JAX configuration.

``-v`` / ``--loglevel`` selects how much detail Python logs, from ``trace``,
``debug``, ``info`` (default), ``warning`` or ``error``. Trace uses level 5.

Sampled output
~~~~~~~~~~~~~~

Request a sampled netCDF history separately from the final snapshot::

   veris run ./my-growth/setup.py --steps 2 --output-snapshot final.nc --output-history history.nc

If the definition supplies ``OutputSettings``, the history keeps its output
streams, calendar, starting date and sampling/averaging schedules. Each stream
is a set of fields saved on a shared schedule. Supplying ``--output-history`` enables a
disabled schedule. When no output settings are defined, the runner creates a
stream that saves current values at every timestep, selecting all fields or
``--io-variables``. A timestep
is one advance of the model clock.

Definitions with enabled scheduled output require ``--output-history``. The final
snapshot and history must have distinct new paths; their parent directories
are created automatically. See :doc:`reference/io` for calendars and averaging.

CPU, GPU and partitioned execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--backend cpu`` or ``--backend gpu`` overrides the definition's backend.
GPU execution requires a compatible installed JAX GPU runtime. A CPU is a
general-purpose processor; a GPU can carry out many calculations in parallel.
The same experiment can run on a local mesh, an arrangement of computing devices
that share the grid. Grid sizes count all physical cells, excluding halos.
The sizes must divide into the requested partitions, with at least two cells
along each axis in each partition. A partition is the part of the grid assigned
to a device. The mesh must contain all selected backend devices::

   JAX_NUM_CPU_DEVICES=2 veris run ./my-island/setup.py --backend cpu --mesh 1 2 --output-snapshot mesh.nc
   JAX_PLATFORMS=cuda,cpu veris run ./my-island/setup.py --backend gpu --output-snapshot gpu.nc

Set logical CPU device counts before starting the process. A ``1 1`` layout
uses serial execution on one device. Distributed runs, which use several Python
processes, use the same explicit JAX rank variables
or multi-task Slurm configuration as :doc:`reference/setups/parallel-dynamics`;
configure a partitioned definition/``--mesh`` and an explicit CPU/GPU backend
before launch. A rank identifies a process in a distributed run. The runner
starts the distributed service and shuts it down on success or failure.

Grouped overrides and shared execution
--------------------------------------

All run commands call :func:`veris.run_experiment`. Named profiles supply the
numerical launch defaults for module drivers. Omitted command line flags preserve
those values. The standalone parallel profile discovers all devices at launch, and
``--steps 0`` writes initialization-only output.

Use ``--set SECTION.KEY=JSON`` for controls without dedicated flags::

   veris run ./my-island/setup.py --steps 2 --set scenario.islandWindSpeed=0.0 --set settings.enable_cyclic_y=false --output-snapshot calm.nc
   python -m veris.setups.run_growth --steps 2 --set forcing.constant.LWdown=300.0 --output-snapshot growth.nc

Mappings, which group values under keys, merge by key. Arrays and stream lists
replace whole values. Values use
JSON, including lowercase ``true``, ``false`` and ``null``. Quote JSON strings,
for example ``--set 'settings.dtype="float32"'``. Unknown controls, malformed
JSON, duplicate paths and overlapping dedicated/generic aliases produce errors
before the runner opens resources or creates output. Overlapping aliases set
the same control through different flags. Use one explicit source for each
control rather than combining ``--steps`` with ``--set steps=...``.

Module snapshots and default histories retain their existing field selections,
calendars and per-step sampling. The runner follows any explicit stream
schedules. The Slurm script supplies small overrides for a quick check. The
shared runtime discovers the mesh and cleans up distributed services.

Inspect and export definitions
-------------------------------

``veris inspect SETUP`` prints a resolved JSON definition: the supplied controls
with their defaults filled in. It does not start JAX devices. Add ``--export PATH``
to create a new file. Add ``--authored`` to keep only the controls explicitly
supplied. You can use the same ``--set SECTION.KEY=JSON`` overrides as run.
The runner protects existing export files. See
:doc:`/quickstart/setup-blueprint` for literal paired installed Python/CLI examples.

Module driver output options
-----------------------------

The ``python -m veris.setups.run_dyn``, ``run_growth`` and ``run_parallel``
drivers accept the output flags below in addition to grouped ``--set`` controls.
Their default sample interval is the model timestep. The central ``veris run``
command instead takes custom schedules from the experiment's ``OutputSettings``;
see :doc:`reference/io` for stream defaults and calendar rules.

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--output-history PATH``
     - History output file.
   * - ``--output-snapshot PATH``
     - Final State snapshot; each driver supplies a default netCDF path.
   * - ``--io-variables hIceMean,Area``
     - Comma-separated State variables for history streams.
   * - ``--sample-seconds NUMBER``
     - Positive time between samples, exactly reachable by model steps.
   * - ``--average instantaneous,daily,monthly,annual``
     - Comma-separated periods, producing independent stream groups.
   * - ``--calendar gregorian|360_day|noleap``
     - Model calendar.
   * - ``--start-date YYYY-MM-DD``
     - Model origin; supports February 30 with ``360_day``.

For example::

   python -m veris.setups.run_growth --steps 30 \
       --output-history growth_history.nc --output-snapshot growth_final.nc \
       --io-variables hIceMean,Area --average instantaneous,monthly \
       --calendar 360_day --start-date 2000-02-01
