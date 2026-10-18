# Vercor scientific validation repeat, 17 October 2026

This record repeats validation of the existing feature source. The accompanying
plain-English explanation is deliberately outside the repository at
`/groups/ocean/nutrik/dev/veris-vercor-integration-explained-2026-10-17.md`.

## Source and scope

- Repository: `https://github.com/team-ocean/veris.git`.
- Tested feature: `4551fc86181c660ceb0010160ec7f569d97e52c6` on
  `feat/vercor-scientific-validation`.
- Pulled comparison branch: `main`,
  `a0a2a7d0c032de42da26c5f09c12b794b2c006b5`.
- Common ancestor: `a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e`.
- All six feature commits are retained. Main was pulled separately, without
  merging its documentation changes into the scientific feature.
- This change adds validation records and corrects one pre-existing runner-test
  backend assumption. A manifest freezes all 251 tracked files under `veris/`,
  `tests/` and `benchmarks/`, including reference archives. Only
  `tests/execution/test_runner.py` changes among those files. Every model,
  benchmark and reference byte remains unchanged. Harness reports say
  `dirty: true`; frozen source hashes separately identify the initial and final
  test versions while documentation is being added.

Both branches were fetched and pulled using the CLI:

```bash
git fetch origin main feat/vercor-scientific-validation
git pull --ff-only origin main
git switch --track origin/feat/vercor-scientific-validation
git pull --ff-only origin feat/vercor-scientific-validation
```

## Runtime and execution policy

Reuse `.venv-latest`, the existing Python 3.14.7 environment. JAX, jaxlib,
`jax-cuda12-plugin` and `jax-cuda12-pjrt` are 0.11.2; NumPy is 2.5.3;
pytest is 9.1.1. CUDA 12 retains Pascal support for these P100 devices.

The local host is `magni.cluster`, with two Tesla P100-PCIE-16GB GPUs and
NVIDIA driver 580.178.04. A preflight explicitly asserts
`jax.default_backend() == "gpu"` and exactly two visible GPU devices. Local
commands run with host access because the sandbox does not expose NVIDIA
devices. Native process/GPU accounting confirms the pytest worker opens both
devices. The GPU suite prefers CUDA while keeping CPU available for tests that
explicitly request it. Such CPU child probes remain CPU checks.

Use eight host CPU cores for the local GPU suite. Run only one pytest session at
a time, with a fresh per-session compilation cache. Full correctness replaces
the ordinary `--fast` development sample. Performance numbers are descriptive
wall times, not a solver benchmark comparison.
The environment snapshot captures the host's 48-core affinity before `taskset`;
the GPU test launches use cores 0–7, as shown below.

## Commands

Artifacts live in the local, untracked directory
`test_logs/vercor-repeat-2026-10-17/`. The existing unrelated `test_logs/`
contents are not staged. Complete scripts, logs, JUnit XML, environment records,
source hashes and machine-readable spherical reports remain there.
The committed JSON evidence embeds the launch and probe-preparation scripts
as text so their exact contents remain available without that local directory.

Local hardware and long-run checks:

```bash
source .venv-latest/bin/activate
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
unset JAX_COMPILATION_CACHE_DIR

# Each run sees one physical GPU; CUDA renumbers that device as cuda:0.
CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python -m benchmarks.validate_spherical \
  --steps 1000 --output test_logs/vercor-repeat-2026-10-17/spherical-gpu-0.json
CUDA_VISIBLE_DEVICES=1 taskset -c 0-7 python -m benchmarks.validate_spherical \
  --steps 1000 --output test_logs/vercor-repeat-2026-10-17/spherical-gpu-1.json

taskset -c 0-7 python -m pytest tests/ -n 1 --dist=loadscope \
  --max-worker-restart=0 -ra \
  --junitxml=test_logs/vercor-repeat-2026-10-17/gpu-full.xml \
  --basetemp=test_logs/vercor-repeat-2026-10-17/gpu-temp
```

Supplemental native two-GPU checks run as standalone Python processes, on host
CPU cores 8–15 while the sole pytest session uses cores 0–7. They are correctness
checks, not an isolated timing comparison. The spherical probe is extracted
from `test_spherical_coordinates_survive_non_square_partitioning` in
`tests/grid/test_spherical_operators.py`; only its two explicit CPU backend
selections change to GPU. The physical-parameter probe uses
`tests/model_physics_sharding_probe.py`; only its three explicit CPU backend
selections and printed labels change. All numerical assertions, finite-difference
steps and tolerances remain unchanged. Parent and generated-source hashes are
recorded in `gpu-probe-provenance.json` and the committed evidence summary.

```bash
# Local preparation script records the exact backend-only adaptations.
python test_logs/vercor-repeat-2026-10-17/prepare_gpu_probes.py
taskset -c 8-15 python test_logs/vercor-repeat-2026-10-17/spherical-two-gpu.py
taskset -c 8-15 python test_logs/vercor-repeat-2026-10-17/physical-parameters-two-gpu.py
# Maintained two-process reduction and reverse-mode derivative oracle:
PYTHONPATH=tests:. taskset -c 8-15 python -c \
  'from reduction_probe import launch_processes; print(launch_processes(platform="cuda"))'
```

The one-node `aegir` job requests the `v3` constraint, 12 CPU cores and 32 GiB
memory. It selects CPU explicitly, uses the normal three-worker policy and
disjoint CPU affinity, and removes SLURM rank variables from the pytest process
so child probes can initialize their own distributed launches. The exact batch
script is `test_logs/vercor-repeat-2026-10-17/full-aegir.slurm`.

```bash
sbatch --hold --parsable test_logs/vercor-repeat-2026-10-17/full-aegir.slurm
# After the local pytest session finishes:
scontrol release 66459424

# Inside the allocation, after activating the existing environment:
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
export COVERAGE_FILE="$PWD/test_logs/vercor-repeat-2026-10-17/.coverage-aegir"
# Execute through srun --cpu-bind=cores, with SLURM rank variables removed:
python -m pytest tests/ -ra --cov=veris \
  --cov-report=json:test_logs/vercor-repeat-2026-10-17/aegir-coverage.json \
  --cov-report=term:skip-covered \
  --junitxml=test_logs/vercor-repeat-2026-10-17/aegir-full.xml \
  --basetemp=test_logs/vercor-repeat-2026-10-17/aegir-temp
coverage report --omit=veris/_version.py --fail-under=80
python -m benchmarks.validate_spherical --steps 1000 \
  --output test_logs/vercor-repeat-2026-10-17/spherical-aegir.json
```

## Results

The original GPU-default full suite completes with **1,733 passed, one failed,
zero skips**, in 5,246.73 s pytest time (5,252.27 s external wall).
The failure is
`tests/execution/test_runner.py::test_public_runner_matches_real_simulation_all_fields_and_diagnostics`:
it expects literal `cpu` metadata although the caller selected the GPU default.
All State and Diagnostics comparisons, clock checks and saved-output assertions
in that test pass before its metadata assertion fails. The identical CPU-only
expectation exists on `main`; the six scientific commits did not introduce it.

`Execution(backend=None)` explicitly honors the caller's JAX choice. Capture
that independently before execution and check the reported metadata against it.
This changes three test lines and no model code, scientific assertion or
numerical tolerance. The corrected real-GPU regression passes in **17.68 s**.
An independent negative-control driver substitutes false CPU metadata for the
real GPU selection; the corrected test rejects it at the backend assertion.
The original full GPU run remains recorded as a failing run; it was not rerun
in full after this metadata-only test correction.

The final CPU suite passes **1,730 tests with four skips**, in **1,320.31 s**
pytest time (**1,349.24 s** external wall). Job **66459424** on **node453**,
partition **aegir**, finishes **COMPLETED, exit 0:0**, using the requested
12 CPUs and 32 GiB allocation. Total job elapsed time is 1,361 s, including the
coverage gate and transport harness. Both full runs collect the same 1,734 case
IDs; the JSON records their matching sorted-case checksum.

| Validation | Passed | Failed | Skipped | External wall time |
| --- | ---: | ---: | ---: | ---: |
| Original GPU-default full suite | 1,733 | 1 | 0 | 5,252.27 s |
| Corrected runner regression on GPU | 1 | 0 | 0 | 20.61 s |
| Final full CPU suite on aegir | 1,730 | 0 | 4 | 1,349.24 s |

The four CPU skips require GPU execution: two central-CLI float32-output cases,
one setup-CLI finite-output case and one two-GPU scheduled-output case. All four
pass in the original GPU-default full run. The CPU result includes the corrected
runner test. Per-domain CPU results are CLI 105 passed/2 skipped, execution
283/0, grid 83/0, I/O 342/1, model 269/0, numerics 168/0, physics 327/0,
setups 82/1 and shared infrastructure 71/0; no domain has a failure.

Statement coverage is **95.07%** for maintained code (4,955/5,212 statements),
excluding generated `veris/_version.py`, and **91.57%** in coverage.py's reported
package scope (5,096/5,565). Standard discovery omits two unexecuted resource
scripts, `veris/cli/templates/dynamics/setup.py` and
`veris/cli/templates/island/setup.py`, with three statements each. Counting all
six as uncovered would give 94.96% maintained coverage. The maintained 80%
coverage gate passes. Maintained Ruff checks,
format checks (235 files), annotation checks, Linux-target type checks and
strict Sphinx also pass; the changed test separately passes Ruff/format/types.

Both native P100s and the aegir CPU pass the 1,000-step transport harness:

| Backend | Maximum relative volume drift | Finite and bounded |
| --- | ---: | --- |
| P100 GPU 0 | 2.220446049250313e-16 | Yes |
| P100 GPU 1 | 2.220446049250313e-16 | Yes |
| aegir CPU | 2.114965734625469e-15 | Yes |

All three report a manufactured spherical-tendency relative error of
5.025120546856513e-13 and tensor refinement orders 1.9997363160 and 1.9999340794.
Each retains intermediate failure status 2, accepts zero host steps after that
failure and returns status 0 for its valid control.

The supplemental two-GPU spherical operators, physical-parameter AD and
two-process reduction/adjoint checks pass. The reduction log retains
coordination-service cancellation/connection-refused messages during shutdown
after both ranks' successful checks; the launcher exits 0.

The committed [JSON evidence](2026-10-17-vercor-repeat.json) contains observed
counts, domain summaries, skip reasons, the original failure traceback,
coverage totals, SLURM accounting, all three spherical reports, launch scripts,
probe provenance and SHA256 checksums of the raw artifacts. Final verification
rehashes all 251 input files: model, benchmark and reference bytes remain
unchanged; only the runner-test backend expectation differs from the tested
feature commit. Independent scientific and validation reviews cover the
rationale, test correction and evidence.

## Interpretation limits

The 1,000-step experiment advances native advection with prescribed velocity.
It checks modeled volume conservation, boundedness and finiteness; separate
manufactured tendency and tensor refinement checks establish spherical metric
accuracy. Rotation RMS error is reported; it has no accuracy acceptance
threshold. This is not a 1,000-step fully coupled ocean–ice trajectory.

The full suite includes independent budget equations, physical-parameter AD and
finite differences, immutable original references with declared v2 budget
corrections, sticky failure status, turning-angle momentum and scheduled-output
regressions. Explicit CPU topology probes do not become GPU tests just because
the parent suite selects CUDA. One-node process checks do not establish
cross-node networking. None of these runs executes an actual Vercor ocean–ice
coupler or establishes observational skill or smooth derivatives across hard
ice-membership transitions.
