Parallel dynamics
=================

The parallel case splits the grid across devices. It shares the dynamics
scenario and timestep for calculating ice motion with the EVP
(elastic-viscous-plastic) method. Its reference default is 1000 steps.
``--nx`` and ``--ny`` are global physical dimensions and must divide evenly
over ``--mesh PX PY``, with at least two cells per partition axis. The mesh
arranges devices along the grid axes and must use all devices of the selected
backend (CPU or GPU). If omitted, it is ``1`` by the number of available
devices. Every partition has two local halo cells on each edge. These extra
cells hold neighboring or boundary data and are exchanged by existing Veris
calculations. Each process allocates only the partitions on devices it can
address.

The CLI defines a dynamics Experiment with the parallel profile and calls
``run_experiment``. The shared runtime checks launch controls, starts services
for communication between processes and constructs the mesh before creating
State, the collection of model arrays. The same public Python workflow
supports coupled physics (motion plus freezing and melting), growth, and
custom ocean geometry/forcing with x/y partitions. Services started by the
framework close when the run ends. Supplied meshes and caller-owned services
stay borrowed, so the caller manages them. The lower-level Simulation
interface requires callers to prepare those resources themselves. Distributed
restart is unsupported. Local CPU tests check splitting arrays across
devices. Actual GPU runs and runs across multiple hosts require suitable
hardware.

Local CPU and local GPU examples::

   JAX_NUM_CPU_DEVICES=4 python -m veris.setups.run_parallel \
       --backend cpu --nx 32 --ny 48 --mesh 2 2 --steps 2 --evp-steps 4 \
       --output-snapshot output/parallel-cpu.nc
   python -m veris.setups.run_parallel --backend gpu --nx 32 --ny 48 \
       --steps 2 --evp-steps 4 --output-snapshot output/parallel-gpu.nc

The CPU scheduler script is ``veris/setups/run_parallel.slurm``. Submit from
the project root::

   sbatch veris/setups/run_parallel.slurm
   sbatch veris/setups/run_parallel.slurm --nx 256 --steps 100 --evp-steps 120 \
       --output-snapshot output/parallel-production.nc

It requests ``aegir``, ``--constraint=v3``, one node, two tasks and four CPU cores
per task. GPUs are used directly on the local node; this script never requests
them. Defaults are a small trial run with a 64-square grid, two steps and four
EVP iterations. Command arguments after the script name override these
defaults. Use sbatch's resource options before the script name to change
nodes/task counts. The default mesh places all ranks (participating processes)
along y. Supply ``--mesh`` for a different arrangement. Log files are
``veris-parallel-JOBID.log`` in the submission directory. The script activates
``.venv-latest`` (or ``.venv`` when that is the installed environment).

Under ``srun``, JAX discovers the SLURM ranks before starting the backend.
Each CPU rank exposes one JAX device and uses Gloo for communication among
processes. For a manual multi-process launch, set
``JAX_COORDINATOR_ADDRESS=HOST:PORT``, ``JAX_NUM_PROCESSES`` and the unique
``JAX_PROCESS_ID`` together for every rank. The CPU environment must be
available on all participating nodes.

For CUDA launches across several Python processes, export
``CUDA_MODULE_LOADING=EAGER`` before starting Python. This avoids delayed kernel
loading during communication; see NVIDIA's
`lazy loading guidance <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html>`_.
When each rank sees one GPU through its own ``CUDA_VISIBLE_DEVICES`` setting,
set ``JAX_LOCAL_DEVICE_IDS=0`` for every rank. The visible GPU must be a
different assigned device for each rank.

Output contains the nine reference fields: hIceMean, Area, hSnowMean, uIce,
vIce, uWind, vWind, uOcean and vOcean. Each partition's halos are removed before
gathering, so arrays have the global physical shape. All ranks help gather
results, and only rank zero writes the netCDF file. Compilation prepares code
without running a model trajectory or collecting output. The reported
integration time includes exactly the requested steps and any scheduled
output. The final snapshot is written after timed integration.

Adding ``--output-history`` enables scheduled output. Chunks of the timestep loop end
at requested instantaneous samples or averaging-window boundaries. Means use
running sums and sample counts on the device, without saving field histories
for every timestep. Unfinished windows pass from one chunk to the next.
Without scheduled output, the runner executes one full-length scan. See
:doc:`../integration` for Python chunk limits and
:ref:`scheduled-output-integration` for collector requirements and how to use
an OutputManager dedicated to writing reduced fields.

Shared scenario and validation
-------------------------------

The physical scenario, changes from the reference, derivative checks and
generated scenario controls are shared with :doc:`dynamics`. This parallel
profile changes the default run length to 1000 steps and adds the launch and
collection behavior above. ``--steps`` overrides the run length.
