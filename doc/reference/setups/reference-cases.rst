Standalone notebook cases
=========================

Three runnable modules reproduce the standalone ``run_dyn.ipynb``,
``run_growth.ipynb`` and ``run_parallel.py`` experiments from
``veris_minimum_working_example`` branch ``jax_halo_exchange`` (reference commit
``2571998f8feb6a5943b214361c122ca78b44c3a8``). They reuse Veris initialization,
frozen State/Configuration/PhysicalConstants and the existing physics kernels.
Importing the modules does not initialize a backend or run a simulation.

Activate the installed environment from the Veris project root::

   source .venv-latest/bin/activate

The commands below select float64 explicitly at runtime. Use ``--help`` for
arguments. Output is a NumPy NPZ archive; the examples need no plotting packages.

Dynamics only
-------------

A 512-km basin has land on its last x and y lines, rotating ocean currents and
the reference's fixed wind snapshot 15 (day 5.875). Wind does not evolve with
time. Defaults are 1024 by 1024 physical cells, 100 steps of 600 seconds and 120
adaptive EVP iterations per step (alpha/beta 500, absolute wind forcing).
There is no thermodynamic growth. A small local CPU run is::

   python -m veris.setups.run_dyn --nx 32 --ny 48 --steps 2 --evp-steps 4 \
       --backend cpu --output output/dynamics.npz

Use ``--backend gpu`` for a GPU directly on the current node. If the environment
already sets ``JAX_PLATFORMS=cpu``, unset it or set ``JAX_PLATFORMS=cuda`` first.
The output contains all final State fields with halos removed, in ``(x, y)``
order. The discarded, synchronized warmup does not count as a simulation step.

The Python API returns independent model objects::

   import jax
   from veris.setups import run_dyn

   jax.config.update("jax_enable_x64", True)
   state, settings, constants = run_dyn.initialize(32, 48)
   state, diagnostics = run_dyn.step_with_diagnostics(state, settings, constants)

``step`` returns just State; ``compiled_step`` compiles the same calculation.
Ocean stresses are separate Diagnostics fields. No coupling output is added to
State. ``settings_overrides``, ``physical_overrides`` and ``scenario_overrides``
allow controlled experiments without changing source files.

Growth only
-----------

The reference column starts with 1.3 m ice, 0.1 m snow, concentration 0.9,
253 K air and 80 W/m2 downward longwave radiation. It executes 150 daily Growth
steps with no dynamics or transport::

   python -m veris.setups.run_growth --backend cpu --output output/growth.npz
   python -m veris.setups.run_growth --backend gpu --output output/growth-gpu.npz

The NPZ ``days`` and ``ice`` arrays hold pre-step samples (days 0 through 149
by default); final State arrays are at day 150. Returned Qnet/Qsw feed the next
step exactly as in the original notebook. They are not reset to initial forcing.
The shared allocator requires at least two cells per interior axis, so the
single-column experiment uses a uniform 2 by 2 interior with two halos per edge;
its pointwise equations are identical to a single column. Final State arrays
include these halos. ``initialize``, ``step``, ``compiled_step`` and
``step_with_diagnostics`` are also available for interactive use and AD.

For example, plot the saved notebook results without requiring Jupyter::

   import numpy as np
   import matplotlib.pyplot as plt

   with np.load("output/growth.npz") as result:
       plt.plot(result["days"], result["ice"])
   plt.xlabel("days")
   plt.ylabel("ice thickness / m")
   plt.show()

Parallel dynamics
-----------------

The parallel case shares the dynamics scenario and timestep; its reference
default is 1000 steps. ``--nx`` and ``--ny`` are global physical dimensions and
must divide evenly over ``--mesh PX PY``, with at least two cells per partition
axis. The mesh must use all devices of the selected backend. If omitted, it is
``1`` by the number of available devices. Every partition has two local halo
cells on each edge, exchanged by existing Veris kernels. Only addressable
partitions are allocated on each process.

Local CPU and local GPU examples::

   JAX_NUM_CPU_DEVICES=4 python -m veris.setups.run_parallel \
       --backend cpu --nx 32 --ny 48 --mesh 2 2 --steps 2 --evp-steps 4 \
       --output output/parallel-cpu.npz
   python -m veris.setups.run_parallel --backend gpu --nx 32 --ny 48 \
       --steps 2 --evp-steps 4 --output output/parallel-gpu.npz

The CPU scheduler script is ``veris/setups/run_parallel.slurm``. Submit from
the project root::

   sbatch veris/setups/run_parallel.slurm
   sbatch veris/setups/run_parallel.slurm --nx 256 --steps 100 --evp-steps 120 \
       --output output/parallel-production.npz

It requests ``aegir``, ``--constraint=v3``, one node, two tasks and four CPU cores
per task. GPUs are used directly on the local node, never requested by this
script. Defaults are a small 64-square, two-step, four-EVP-iteration smoke run;
trailing command arguments override these. Use sbatch's resource options before
the script name to change nodes/task counts. The default mesh spans all ranks
along y; supply ``--mesh`` for a different topology. Log files are
``veris-parallel-JOBID.log`` in the submission directory. The script activates
``.venv-latest`` (or ``.venv`` when that is the installed environment).

Under ``srun``, JAX discovers the SLURM ranks before backend initialization.
Each CPU rank exposes one JAX device and uses Gloo collectives. For a manual
multi-process launch, set ``JAX_COORDINATOR_ADDRESS=HOST:PORT``,
``JAX_NUM_PROCESSES`` and the unique ``JAX_PROCESS_ID`` together for every rank.
The CPU environment must be available on all participating nodes.

Output contains the nine reference fields: hIceMean, Area, hSnowMean, uIce,
vIce, uWind, vWind, uOcean and vOcean. Each partition's halos are removed before
gathering; arrays have the global physical shape. All ranks participate in the
gather, and only rank zero writes the archive. Warmup is synchronized and
discarded; the reported integration time includes exactly the requested steps.

Reference adaptations
---------------------

The runnable cases correct the old serial mask's five-cell neighbor offset,
use the physical one-cell face stencil, and initialize periodic halos
consistently. They replace generated initializer imports, fixed home paths and
all-rank output writes. Fields unused by each isolated experiment retain the
shared VARIABLES defaults rather than the old blanket zero initialization;
all prescribed fields consumed by the experiment retain their reference values.
Experimental constants live in scenario registries, while universal physical
constants stay in PhysicalConstants. No changes to the physics kernels are needed.

Derivative validation
---------------------

The exact dynamics initial state has zero strain and closed walls. Existing EVP
norm derivatives can produce NaN in reverse mode at that state, even when the
forward solution is finite. The new driver preserves those equations; it does
not regularize this singularity. Case-level derivative tests use the established
smooth, nonuniform open-ocean fixture and compare against finite differences.
The reference growth column also has a finite-difference-checked longwave
sensitivity. These checks do not imply differentiability across every clipping
threshold or zero-norm state.

Scenario registries
-------------------

.. exec::

   from veris.setups.run_dyn import DYNAMICS_SETTINGS
   from veris.setups.run_growth import GROWTH_SETTINGS
   for title, registry in (("Dynamics", DYNAMICS_SETTINGS), ("Growth", GROWTH_SETTINGS)):
       print(f".. list-table:: {title} scenario controls")
       print("   :header-rows: 1\n")
       print("   * - Setting\n     - Default\n     - Units\n     - Description")
       for name, metadata in registry.items():
           print(f"   * - ``{name}``\n     - ``{metadata.default!r}``")
           print(f"     - {metadata.units}\n     - {metadata.description}")
       print("")
