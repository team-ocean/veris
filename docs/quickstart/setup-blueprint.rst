Add and reproduce an experiment
================================

A setup loads data in ordinary Python code (on the host) and returns
:class:`veris.Experiment`. Veris fills in settings, checks physical limits,
builds the grid for a serial run or the regions assigned to the current process,
and runs the shared physics. It also prepares halos, extra cells at grid edges,
according to each field's boundary rules. Keep device selection (choosing
processors), distributed bootstrap (starting coordination between processes)
and integration (advancing the model through time) out of ``build_experiment``.

Start with prescribed ocean geometry
------------------------------------

Copy the following file to ``setup.py``. The grid divides the modeled area into
cells. Its dimensions count physical cells across the whole grid; Veris supplies
two halos at every partition edge, where the grid is divided between devices.
Ocean masks mark ocean and land. They have shape ``(nx, ny, z)`` with the surface
at the last level. Masks lie in [0, 1]; spacing/area are positive; temperatures
use kelvin. Ice/snow thickness is nonnegative and concentration, the fraction
of each cell covered by ice, lies in [0, 1].

.. literalinclude:: ../examples/framework/setup.py
   :language: python

Run from Python
----------------

The runner applies explicit overrides last, after shared defaults, preset
defaults and experiment configuration values. It creates missing output
directories after preflight (checks before execution) and protects existing
files::

   from pathlib import Path
   from veris import run_experiment
   from setup import build_experiment

   result = run_experiment(
       build_experiment(),
       overrides={"steps": 3, "execution": {"backend": "cpu"}},
       snapshot_path=Path("python.nc"),
   )

Run from the CLI
----------------

The same experiment definition and overrides produce the same 70 physical
fields, or arrays of model values::

   veris run setup.py --steps 3 --backend cpu --output cli.nc

Inspect and export
-------------------

Inspection checks inputs in ordinary Python without configuring JAX (the
library for array calculations) or asking which devices are available::

   veris inspect setup.py --export resolved.json
   veris inspect setup.py --authored --export authored.json

Run either JSON export (an experiment definition stored in a text file) through
the same command-line interface (CLI) or Python loader. A resolved export saves
all the effective values, fixing them for later use. An authored export keeps
the inputs you wrote, so omitted values still inherit defaults. See
:ref:`experiment-export-modes` for the exact file format and rules for combining
settings. Existing export paths are protected.

A compact authored growth example is included alongside its
``growth-resolved.json`` export, which saves all the effective values:

.. literalinclude:: ../examples/framework/growth-authored.json
   :language: json

Defaults and overrides
-----------------------

Use the grouped settings and the override order described in
:doc:`/api/experiment`. Dedicated CLI options and their grouped aliases must not
overlap; see :doc:`/cli` for syntax.

For an existing setup, use ``Experiment.builtin("growth")`` or a documented
profile (a named set of defaults), such as
``Experiment.builtin("dynamics", profile="standalone")``. Presets keep their
intentional differences in starting values. All use the current shared
``advance``/``step`` kernels, the functions that perform the model calculations.

Register a field factory
------------------------

A reusable setup can register :class:`~veris.setups.presets.SetupPreset` once
per process. Its scenario registry lists the physical defaults and metadata
(information about each setting). ``scenario_factory(values, dtype)`` fills in
and checks values against those rules. Use the existing
``registry_defaults``/``validate_scalars`` helpers for scalar controls (settings
with single values). The regional field factory, a function that supplies
values for one region of the grid, has this signature::

   def fields(state, configuration, constants, scenario, global_shape, x, y, *, regional):
       return {"hIceMean": scenario["initial_ice"], "Area": 0.8}

It returns replacements for fields in State, the arrays holding the model's
values. Use shapes for the local region including its halos, or scalars (single
values). ``x``/``y`` locate the region's physical cells in the whole grid;
``global_shape`` excludes all halos. Veris allocates arrays, applies inputs,
calculates ice load (mass per area) and inverse thickness, and handles the outer
grid boundaries. Declare ``initial_defaults`` for physical inputs derived from
the scenario. This lets Veris check their limits and numerical precision before
selecting resources. New providers do not need name-specific branches in the
core code. See :doc:`/api/initialization` and its tested registry rules.

Preset numerical defaults may select ``dtype``, the numerical type and precision.
Profile and explicit settings override it before Veris puts scenario values
into their final form (normalization). Defaults supplied by functions must
return a dtype that stays the same through that process. If resolving defaults
repeatedly changes the dtype in a cycle, definition validation fails; an explicit
dtype can settle the choice. Veris stores its own immutable copies of authored
scenario arrays and nested controls, so changing the originals cannot change
the definition.

Local and distributed execution
--------------------------------

Set logical device counts (the number of devices JAX presents) before starting
Python. The existing setup can run on a local CPU mesh, a group of computer
processor devices on one machine, or on the available graphics processors
(GPUs)::

   JAX_NUM_CPU_DEVICES=2 veris run setup.py --backend cpu --mesh 1 2 --output cpu-mesh.nc
   JAX_PLATFORMS=cuda,cpu veris run setup.py --backend gpu --mesh 1 2 --output gpu-mesh.nc

The requirements for meshes and runs across processes are defined in
:doc:`/api/parallel`. Explicit ``DistributedExecution`` controls override the
JAX/Slurm launch environment, the variables set when starting the processes.

To run across Python processes, request the same execution values on every
process::

   from pathlib import Path
   from veris import Execution, run_experiment
   from setup import build_experiment

   definition = build_experiment().with_overrides({
       "execution": {"backend": "cpu", "partitions": (1, 2)},
   })
   result = run_experiment(definition, snapshot_path=Path("distributed.nc"))

Start those processes with explicit JAX rank variables, which identify each
process, or an ``srun`` allocation from Slurm, a cluster job scheduler. The runner
manages and closes the resources it starts. :doc:`/api/parallel` describes
borrowed resources (ones the caller already started) and input checks. The
low-level Simulation interface accepts resources that are already prepared.

The maintained smoke launcher, for a short verification run, uses the CPU
``aegir`` partition (a cluster resource group). It uses the parallel profile's
discovery of all devices::

   sbatch veris/setups/run_parallel.slurm --output aegir.nc

For two-node verification, use two tasks with one task per node (cluster
computer). All ranks (processes) must participate in collectives, operations
that exchange data across processes, and use the same output destinations.
Only the writing rank creates the output file. See
:doc:`/reference/setups/parallel-dynamics` for launch details.
