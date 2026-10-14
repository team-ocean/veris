Command line tools
==================

After installing Veris, the ``veris`` command is available from any directory::

   python -m pip install -e .
   veris --help
   veris --version

Reinstall an existing checkout to register the console command. You can also
use ``python -m veris.cli`` with the same arguments. The central interface uses
Click and follows the discovery, copy and run workflow of the
`Vercor command line tools <https://vercor.readthedocs.io/en/latest/cli.html>`_.
It executes Veris's existing :doc:`Experiment and Simulation lifecycle
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

The builtin templates are ``island`` (coupled dynamics, transport and growth
with prescribed ocean masks and fields), ``dynamics`` (a fixed-wind basin) and
``growth`` (a prescribed thermodynamic column). See :doc:`reference/setup-gallery`
for their physical assumptions and numerics. All use the maintained preset
initialization; copying a template does not run or allocate a model.

Copy a selected setup
---------------------

Copy a template and its relative assets to an editable directory::

   veris copy-setup island --to ./my-island

The destination must be new. Without ``--to``, the destination is ``./SETUP``
in the current directory. Parent directories are created as needed. Existing
directories, files and symlinks are rejected, retaining any user edits.

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
   veris run my-basin --backend cpu --steps 3 --output basin.nc

Names are sorted and must be unique across builtin and external roots.
Missing roots and duplicate names produce errors. Discovery inspects only
these directories and does not import their Python files.

To create your own setup outside the Veris repository, copy a builtin under
your chosen name into an external root, then edit its ``setup.py``::

   veris copy-setup island --to /path/to/extra-setups/my-sea-ice
   export VERIS_SETUP_DIR=/path/to/extra-setups
   veris show-setups
   veris inspect my-sea-ice
   veris run my-sea-ice --backend cpu --steps 3 --output my-sea-ice.nc

The folder name is the CLI setup name; avoid names ending in ``.py`` or
``.json``, which identify files. Set ``VERIS_SETUP_DIR`` in each shell that
should discover it. An existing file takes precedence over name lookup;
directories and dangling symlinks are rejected. Explicit paths (including
``./`` prefixes) and missing ``.py``/``.json`` files never fall back to a setup
name. Named runs use the same factory, overrides and output behavior as file
runs. Builtin names can also be run directly, for example ``veris run growth``.

Run a setup
-----------

Run a copied Python setup, optionally overriding execution controls::

   veris run ./my-island/setup.py --backend cpu --steps 3 --output island.nc

The following complete example copies and runs the thermodynamic column::

   veris copy-setup growth --to ./my-growth
   veris run ./my-growth/setup.py --steps 2 --float-type float32 --output growth.nc

Output paths are relative to the current working directory. A final snapshot
defaults to ``output.nc`` and contains all native State fields at the final
absolute model time, with halos removed. Use fresh destinations for repeated
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

A Python setup defines a zero-argument ``build_experiment`` function returning
a validated ``Experiment``. For example::

   from veris import Execution, Experiment, Grid

   def build_experiment():
       return Experiment.builtin(
           "island", grid=Grid(8, 12), steps=3,
           settings={"dtype": "float32", "nEVPsteps": 5},
           execution=Execution(backend="cpu"),
       )

Edit the definition to supply numerical settings, physical constants, initial
conditions, forcing, geometry or output schedules. The factory constructs host
definitions and leaves device allocation and execution to the runner. Python
setup files execute trusted user code. Their sibling helper modules can be
imported; locate input assets relative to ``Path(__file__).parent``. The runner
keeps the caller's working directory and does not execute an
``if __name__ == "__main__"`` block. Existing standalone setup drivers remain
available through their documented ``python -m veris.setups...`` commands.

JSON and overrides
~~~~~~~~~~~~~~~~~~

The same runner accepts :doc:`versioned JSON definitions <api/experiment>`::

   veris run experiment.json --backend cpu --output final.nc

File values take precedence over registry/preset defaults. Explicit CLI flags
then override steps, dtype, backend and mesh; omitted flags preserve the file's
values. Float64 is the model default. ``--float-type`` updates both numerical
and physical-constant precision. The runner enables JAX x64 before allocation,
including the float64 reductions used by sampled output with float32 physics.
Help, version, discovery and copy commands do not change JAX configuration.

``-v`` / ``--loglevel`` selects Python logging verbosity from ``trace``,
``debug``, ``info`` (default), ``warning`` or ``error``. Trace uses level 5.

Sampled output
~~~~~~~~~~~~~~

Request a sampled netCDF history separately from the final snapshot::

   veris run ./my-growth/setup.py --steps 2 --output final.nc --netcdf history.nc

If the definition supplies ``OutputSettings``, the history keeps its streams,
calendar, origin and sampling/averaging schedules, enabling a disabled schedule
when ``--netcdf`` is supplied. Otherwise the runner creates
a per-timestep instantaneous stream, selecting all fields or ``--io-variables``.
Definitions with enabled scheduled output require ``--netcdf``. The final
snapshot and history must have distinct new paths; their parent directories
are created automatically. See :doc:`reference/io` for calendars and averaging.

CPU, GPU and partitioned execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--backend cpu`` or ``--backend gpu`` overrides the definition's backend.
GPU execution requires a compatible installed JAX GPU runtime. The same
experiment can run on a local mesh; grid sizes count global physical cells and
must divide into the requested partitions with at least two cells per axis
per partition. The mesh must contain all selected backend devices::

   JAX_NUM_CPU_DEVICES=2 veris run ./my-island/setup.py --backend cpu --mesh 1 2 --output mesh.nc
   JAX_PLATFORMS=cuda,cpu veris run ./my-island/setup.py --backend gpu --output gpu.nc

Set logical CPU device counts before starting the process. A ``1 1`` layout
uses serial execution. Distributed runs use the same explicit JAX rank variables
or multi-task Slurm configuration as :doc:`reference/setups/parallel-dynamics`;
configure a partitioned definition/``--mesh`` and an explicit CPU/GPU backend
before launch. The runner owns
distributed initialization and shuts it down on success or failure.

Grouped overrides and shared execution
--------------------------------------

All run commands call :func:`veris.run_experiment`. Numeric module-driver
launch defaults come from named profiles; omitted parser flags preserve those
values. The standalone parallel profile discovers all devices at launch, and
``--steps 0`` writes initialization-only output.

Use ``--set SECTION.KEY=JSON`` for controls without dedicated flags::

   veris run ./my-island/setup.py --steps 2 --set scenario.islandWindSpeed=0.0 --set settings.enable_cyclic_y=false --output calm.nc
   python -m veris.setups.run_growth --steps 2 --set forcing.constant.LWdown=300.0 --output growth.nc

Mappings merge by key; arrays and stream lists replace whole values. Values use
JSON, including lowercase ``true``, ``false`` and ``null``. Quote JSON strings,
for example ``--set 'settings.dtype="float32"'``. Unknown controls, malformed
JSON, duplicate paths and overlapping dedicated/generic aliases fail before
resource acquisition or output creation. Use one explicit source for each
control rather than combining ``--steps`` with ``--set steps=...``.

Module snapshots and default histories retain their existing field selections,
calendars and per-step sampling. Explicit stream schedules remain authoritative.
The Slurm script supplies deliberate small smoke overrides while the shared
runtime owns mesh discovery and distributed service cleanup.

Inspect and export definitions
-------------------------------

``veris inspect SETUP`` prints a resolved JSON definition without launching JAX
devices. Add ``--export PATH`` to create a new file, ``--authored`` to retain
sparse authored controls, and the same ``--set SECTION.KEY=JSON`` overrides used
by run. Existing export files are protected. See
:doc:`/quickstart/setup-blueprint` for literal paired installed Python/CLI examples.
