Add and reproduce an experiment
================================

A setup provides host data and returns :class:`veris.Experiment`. The framework
resolves controls, validates physical bounds, builds serial/addressable regions,
applies field-aware halos and executes the shared physics. Keep device selection,
distributed bootstrap and integration out of ``build_experiment``.

Start with prescribed ocean geometry
------------------------------------

Copy the following file to ``setup.py``. Grid dimensions count global physical
cells; the framework supplies two halos at every partition edge. Ocean masks
have shape ``(nx, ny, z)`` with the surface at the last level. Masks lie in
[0, 1]; spacing/area are positive; temperatures use kelvin. Ice/snow thickness
is nonnegative and concentration lies in [0, 1].

.. literalinclude:: ../examples/framework/setup.py
   :language: python

Run from Python
----------------

The runner applies explicit overrides after shared/preset/configuration values.
It creates missing output parents after preflight and protects existing files::

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

The same specification and overrides produce the same 70 physical fields::

   veris run setup.py --steps 3 --backend cpu --output cli.nc

Inspect and export
-------------------

Inspection validates host inputs without configuring JAX or enumerating devices::

   veris inspect setup.py --export resolved.json
   veris inspect setup.py --authored --export authored.json

Run either JSON export through the same CLI/Python loader. Use a resolved
export to pin effective values, or an authored export to preserve inherited
defaults. See :ref:`experiment-export-modes` for the precise schema and merge
contracts. Existing export paths are protected.

A compact authored growth example is included alongside its fully pinned
``growth-resolved.json`` export:

.. literalinclude:: ../examples/framework/growth-authored.json
   :language: json

Defaults and overrides
-----------------------

Use the grouped controls and precedence described in :doc:`/api/experiment`.
Dedicated and grouped CLI aliases must not overlap; see :doc:`/cli` for syntax.

For an existing setup, use ``Experiment.builtin("growth")`` or a documented
profile, such as ``Experiment.builtin("dynamics", profile="standalone")``.
Presets retain their intentional startup differences and all use the current
shared ``advance``/``step`` kernels.

Register a field factory
------------------------

A reusable setup can register :class:`~veris.setups.presets.SetupPreset` once
per process. Its scenario registry owns scientific defaults and metadata;
``scenario_factory(values, dtype)`` resolves/validates that schema. Use the
existing ``registry_defaults``/``validate_scalars`` helpers for scalar controls.
The regional field factory has this signature::

   def fields(state, configuration, constants, scenario, global_shape, x, y, *, regional):
       return {"hIceMean": scenario["initial_ice"], "Area": 0.8}

It returns State field replacements using local halo-inclusive shapes or
scalars. ``x``/``y`` are global physical indices for this addressable region;
``global_shape`` excludes all halos. The framework owns allocation, input
application, derived load/inverse thickness and global boundaries. Declare
``initial_defaults`` for scenario-derived physical inputs so bounds/precision
can be checked before resource selection. New providers need no core name
branches. See :doc:`/api/initialization` and its tested registry contract.

Preset numerical defaults may select ``dtype``; profile and explicit settings
override it before final scenario normalization. Callable defaults must return
a dtype that is stable under that normalization. Cycles fail during definition
validation; an explicit dtype can settle the choice. Authored scenario arrays
and nested controls are owned immutable copies.

Local and distributed execution
--------------------------------

Set logical device counts before starting Python. The existing setup can run
on a local CPU mesh or the available GPUs::

   JAX_NUM_CPU_DEVICES=2 veris run setup.py --backend cpu --mesh 1 2 --output cpu-mesh.nc
   JAX_PLATFORMS=cuda,cpu veris run setup.py --backend gpu --mesh 1 2 --output gpu-mesh.nc

Mesh and distributed resource requirements are defined in
:doc:`/api/parallel`. Explicit ``DistributedExecution`` controls override the
JAX/Slurm launch environment.

For distributed Python, request the same execution values on every process::

   from pathlib import Path
   from veris import Execution, run_experiment
   from setup import build_experiment

   definition = build_experiment().with_overrides({
       "execution": {"backend": "cpu", "partitions": (1, 2)},
   })
   result = run_experiment(definition, snapshot_path=Path("distributed.nc"))

Launch those processes with explicit JAX rank variables or an ``srun`` allocation.
The runner owns the resources it starts; :doc:`/api/parallel` describes borrowed
resources and validation. The low-level Simulation facade accepts already
prepared resources.

The maintained smoke launcher uses the CPU ``aegir`` partition and the parallel
profile's all-device discovery::

   sbatch veris/setups/run_parallel.slurm --output aegir.nc

For two-node acceptance, use two tasks with one task per node; all ranks must
participate in collectives and use the same output destinations. Only the
writing rank creates the output file. See
:doc:`/reference/setups/parallel-dynamics` for launch details.
