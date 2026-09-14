# Standalone reference cases implementation plan

Goal: Adapt run_dyn.ipynb, run_growth.ipynb and run_parallel.py from the
jax_halo_exchange reference branch into separate runnable veris/setups cases.

Architecture: Reuse frozen State, Configuration, PhysicalConstants and shared
initialization; compose existing physics kernels. Serial dynamics and parallel
execution share one dynamics step and reference scenario. Growth remains an
independent uniform-column experiment. Runtime selection precedes JAX backend
initialization. Outputs and coupling diagnostics remain outside State.

Constraints: current jax-only checkout; no changes to reference repositories.
CPU SLURM jobs use aegir and constraint v3; GPUs run on the local node. Preserve
fixed snapshot 15 winds and reference physics order. Keep source defaults
(1024-square, 100 dynamics steps; 150 growth days; 1000 parallel steps), expose
small runs through CLI. Correct halo layout and mask orientation for current
Veris x/y convention. No generated imports or machine-specific home paths.

- [x] Root: reference dynamics initialization, step, runner and equation/sequence tests.
- [x] Growth agent: growth initialization, step, runner and reference sequence tests.
- [x] Root: parallel CPU/GPU mesh execution, local partition allocation, gathering,
      SLURM environment/bootstrap and aegir batch script.
- [x] Validate serial/sharded equivalence on multiple CPU devices and processes,
      GPU smoke runs, SLURM execution, CLI output, invalid arguments.
- [x] Independent code/test review, documentation and CHANGELOG updates.
- [x] Full correctness suite and maintained lint/type checks before commit.

Ruling: User request already authorizes implementation; proceed with reviewable
files and tests. Use installed .venv-latest (the only existing environment),
without recreating dependencies. Root owns pytest scheduling: one instance only.
