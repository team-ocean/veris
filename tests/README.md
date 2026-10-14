# Numerical tests

See [AUDIT.md](AUDIT.md) for the redundancy audit and test ownership rules.
Add assertions to an existing run when inputs and trajectories are identical;
retain separate tests for distinct physical branches, precision and AD contracts.

Tests mirror the [scientific package hierarchy](../docs/reference/package-hierarchy.rst):

| Directory | Test ownership |
| --- | --- |
| `model/` | State and array contracts, settings, constants, definitions, forcing and metadata |
| `grid/` | Geometry, ocean initialization, interpolation, halos, reductions and regional inputs |
| `physics/dynamics/` | Mass, rheology, drag, ocean stress and dynamics composition |
| `physics/transport/` | Advection, cleanup and ridging |
| `physics/thermodynamics/` | Growth, bulk heat fluxes and their gradients |
| `numerics/` | Solvers, differentiable primitives, precise equations and gradient checks |
| `execution/` | Initialization, pure timesteps, rollouts, lifecycle and resource ownership |
| `io/` | Scheduled output, observations, storage, selected input and complete restart |
| `setups/` | Preset initialization and command-line examples |

Domain tests exercise the canonical owners and numerical contracts. Shared
`conftest.py`, standalone process probes, immutable `reference_data/`, packaging,
benchmark and runtime tests remain at the test root. Schema tests verify the
single owners of State, array inputs and OceanGeometry and exercise cold imports.

Activate `.venv-latest` before commands. Install the reproducible CPU environment
with `module load uv/latest` and `uv pip install -r requirements-test.txt`.
The repository lock targets Python 3.14 with JAX and jaxlib 0.11.2, matching
GitHub CI and the local `.venv-latest`. `requirements-gpu.txt` adds the matching
CUDA 12 plugin for the local GPUs. Create the environment with
`uv venv --python 3.14 .venv-latest` and activate it before installation.

Run `pytest tests/ --fast` during development. `VERIS_TEST_SEED` varies the stable
10% collection sample; the default is `root`. Run the full suite before commits:

```sh
JAX_PLATFORMS=cpu pytest tests/ -n 3 --dist=loadscope --partition-cpus --max-worker-restart=0 --cov=veris --cov-report=term:skip-covered
coverage report --omit=veris/_version.py --fail-under=80
ruff check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py
ruff format --check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py
ruff check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py --select ANN001,ANN002,ANN003,ANN201,ANN202,ANN204,ANN205,ANN206
ty check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py
```

For focused gradients, use `pytest tests/numerics/test_gradients.py`.
Run one coordinated pytest session at a time. The full command uses three xdist
workers, grouped by module to retain shared fixtures and compilation reuse.
`--partition-cpus` gives each worker a disjoint strided group from the process's
permitted CPUs before collection or JAX backend initialization; child probes
inherit that placement. Disable worker restart so crashes fail the full run.
Use fewer workers if fewer CPUs are available. CPU CI uses one worker so the
four-device sharding probes inherit all four permitted runner CPUs; splitting
them among three workers can stall XLA collectives even on JAX 0.11.2.
The full correctness step has no explicit 20-minute timeout.
Dated correctness counts, coverage and elapsed times live in
[CHANGELOG.md](../CHANGELOG.md), alongside the corresponding source revisions.
Keep solver performance measurements in the separate
[benchmark harness](../benchmarks/README.md).

Each worker starts with a fresh private JAX compilation-cache directory and
shares it with its child probes. Worker caches are separate because unbounded
JAX disk-cache writes are not locked. When the caller has no cache directory or explicit
compile-time policy and JAX retains its default one-second threshold, the owned
session sets the minimum compile time to zero so subsecond compilations can be
reused. Children inherit that policy through the environment. Caller-owned cache
paths, disabled caches, explicit environment thresholds, nondefault configured
thresholds and the entry-size policy are preserved. Teardown resets the owned
cache, restores configuration and environment settings, and removes its directory
on success or failure. Each full session starts cold; no preceding session's
cache is required. `test_pytest_runtime.py` owns these lifecycle regressions.

Tests execute actual compiled JAX functions on small rectangular grids,
with precision chosen by the assertion's contract:

- Run validation, metadata, shape, mask and indexing checks once. These tests
  normally use the default dtype; do not add a precision Cartesian product.
- Use float64 for independent equations and precise JVP/VJP/finite-difference
  references, including distributed gradient comparisons.
- Use float32 for representative operational rollouts, restart equivalence and
  overflow/stability checks.
- Test both only for dtype propagation, conversion, precision-sensitive
  thresholds and the small coupled integration sample. Scheduled-output tests
  retain both because they check float64 accumulation with each physics dtype
  and preservation of the global x64 policy.

Keep dtype assertions explicit in precision tests. A structural assertion used
inside a numerical comparison (for example, equal leaf shapes before comparing
values) is still required to prevent broadcasting from hiding a failure.

Expected results come from explicit neighborhood indexing, scalar threshold
rules, momentum balance, and finite differences. Nonsmooth tests check selected
AD linearizations and one-sided slopes separately.

Whole-package reports include generated version metadata. With user approval,
CI enforces at least 80% coverage of maintained code by excluding only
`veris/_version.py` from the gate. XML/JSON artifacts retain whole-package data.
The old geographic setup and its ocean-model dependencies were removed at the
user’s request. CPU CI is enforced. Local GPU requirements are recorded in
`requirements-gpu.txt`; see [CHANGELOG.md](../CHANGELOG.md) for dated hardware
and native GPU verification results. CPU correctness and targeted native GPU
checks have distinct scopes.

Current physical coverage includes periodic transport, rheology, wind and ocean
stress, EVP momentum limits, and thermodynamic energy/water budgets. Serial
mode uses `Configuration(use_sharding=False)` when calling halo
consumers. Fresh-process tests verify real four-CPU-device halo exchange and
reverse-mode sensitivities on 2x2, 1x4 and 4x1 meshes. Separate reduction probes
verify two-process CPU and GPU collectives. These local-machine checks do not
certify multi-node networking or MPI launchers.

The [Python lifecycle tutorial](../docs/quickstart/user-guide.rst) provides the
standalone island run. Test harness usage lives here; beginner installation and
first execution live in [Getting started](../docs/introduction/get-started.rst).

Hardware validation uses `requirements-gpu.txt` for the local Tesla P100s.
Require CUDA explicitly while retaining CPU devices for diagnostic callbacks:

```bash
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -c 'import jax, pytest; assert jax.default_backend() == "gpu"; raise SystemExit(pytest.main(["tests/", "-q"]))'
```

`grid/test_global_sum.py` launches two real CPU processes. To exercise the same
reduction and adjoint oracles on one GPU per process:

```bash
PYTHONPATH=tests:. JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -c 'from reduction_probe import launch_processes; print(launch_processes(platform="cuda"))'
```

Nonsmooth gradient tests distinguish selected JAX linearizations from classical
one-sided slopes. In particular, thin-ice removal is discontinuous and its
zero branch derivative cannot represent crossing the removal threshold.

Inside `jax.shard_map`, pass `axis_names=("x", "y")` to `IceVelocities` or
`evp_solver` to obtain global residual diagnostics. Callers supply local blocks
with two-cell halos; norms exclude those halos before collective reduction.
The default empty tuple preserves serial behavior. The uniform sharded EVP
test isolates reduction semantics; it is not a full distributed coupled run.

The public lifecycle tests cover definition and JSON validation, custom ocean
geometry/masks, initialization, pure timesteps, successive host runs, forcing
position, scheduled means, snapshots, selected input and complete serial restart.
Frozen original all-field trajectories protect preset and numerical-kernel behavior;
their provenance and reproduction instructions are in
[reference_data/README.md](reference_data/README.md).

`execution/test_lifecycle_sharding.py` runs real four-CPU-device experiments in fresh
processes. Its probe covers presets and custom geometry, both boundary modes,
2x2/1x4/4x1 layouts, continued forcing/output and two-cell local extents.
Separate pure, rollout and closed-wall probes retain JIT, grad, JVP, VJP and
finite-difference checks. Serial resource tests exercise initialization and
execution inside an ambient mesh while checking single-device placement and
restoration of the caller's context. These tests cannot certify actual
multi-host initialization, networking, file ownership or failure cleanup.

CI also builds documentation with warnings treated as errors, builds fresh
wheel/source archives, installs the wheel in a clean target, and executes literal
public examples from the quickstart and README outside the source import path.
`installed_lifecycle_probe.py` verifies package origin, coupled save/resume,
selected input, custom ocean fields, two-CPU sharding and pure JIT/AD examples.
`test_packaging.py` checks that the source archive includes the complete test
harness, benchmark helpers, reference data, lifecycle design and parallel launcher.

`grid/test_halo_compilation.py` verifies shared halo tracing with changing values,
separate shape/dtype compilations and independent slice/JVP/VJP expectations.
`io/test_compilation_reuse.py` checks the stable scheduled executor with successive
forcing values and independent output managers. `io/test_callback_lifetime.py`
checks arbitrary callable support, observer overrides and collection of captured
resources and JIT owners. `test_parallel_runtime.py` covers CPU placement and
private worker-cache lifecycles, including one-CPU and unsupported-platform
contracts. See the verified [reorganization audit](AUDIT.md).
