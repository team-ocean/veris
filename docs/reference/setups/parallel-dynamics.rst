Parallel dynamics
=================

The parallel case shares the dynamics scenario and timestep; its reference
default is 1000 steps. ``--nx`` and ``--ny`` are global physical dimensions and
must divide evenly over ``--mesh PX PY``, with at least two cells per partition
axis. The mesh must use all devices of the selected backend. If omitted, it is
``1`` by the number of available devices. Every partition has two local halo
cells on each edge, exchanged by existing Veris kernels. Only addressable
partitions are allocated on each process.

The CLI defines a dynamics Experiment with the parallel profile and calls
``run_experiment``. The shared runtime validates launch controls, initializes
distributed services and constructs the mesh before State allocation. The same
public Python workflow supports coupled, growth and custom ocean geometry/forcing
with x/y partitions. Framework-owned services close when the run ends; supplied
meshes and caller-owned services remain borrowed. The lower-level Simulation
facade requires callers to prepare those resources explicitly. Distributed
restart is unsupported. Local CPU tests validate sharding; actual GPU and
multihost runs require suitable hardware.

Local CPU and local GPU examples::

   JAX_NUM_CPU_DEVICES=4 python -m veris.setups.run_parallel \
       --backend cpu --nx 32 --ny 48 --mesh 2 2 --steps 2 --evp-steps 4 \
       --output output/parallel-cpu.nc
   python -m veris.setups.run_parallel --backend gpu --nx 32 --ny 48 \
       --steps 2 --evp-steps 4 --output output/parallel-gpu.nc

The CPU scheduler script is ``veris/setups/run_parallel.slurm``. Submit from
the project root::

   sbatch veris/setups/run_parallel.slurm
   sbatch veris/setups/run_parallel.slurm --nx 256 --steps 100 --evp-steps 120 \
       --output output/parallel-production.nc

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
gather, and only rank zero writes the netCDF file. Compilation executes no
trajectory or collection. The reported integration time includes exactly the
requested steps and any scheduled output. Final snapshot writing follows the
timed integration.

Adding ``--netcdf`` enables scheduled output. Scan chunks end at requested
instantaneous samples or averaging-window boundaries. Means use device running
sums and sample counts, without storing timestep field histories; unfinished
windows carry across chunks. Without scheduled output, the runner executes one
full-length scan. See :doc:`../integration` for Python chunk caps and
:ref:`scheduled-output-integration` for collector requirements and the exclusive
reduced-writing OutputManager lifecycle.

Shared scenario and validation
-------------------------------

The physical scenario, reference adaptations, derivative checks and generated
scenario controls are shared with :doc:`dynamics`. This parallel profile changes
the default run length to 1000 steps and adds the launch and collection behavior
above; ``--steps`` overrides the run length.
