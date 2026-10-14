# Numerical tests

See [AUDIT.md](AUDIT.md) for the redundancy audit and test ownership rules.
Add assertions (checks of expected results) to an existing run when its inputs
and trajectories (model values through time) are identical. Keep separate tests
for different physical branches, numerical precision (how finely numbers are
represented) and AD contracts. AD means automatic differentiation, calculating
how changes in inputs affect results; its contracts specify the required behavior.

Tests mirror the [scientific package hierarchy](../docs/reference/package-hierarchy.rst):

| Directory | Test ownership |
| --- | --- |
| `model/` | Rules for State (the arrays holding model values) and array inputs; settings, constants, experiment definitions, forcing (supplied inputs) and metadata (information about each value) |
| `grid/` | Geometry (grid shape and sizes), preparing ocean inputs, interpolation (estimating values between grid locations), halos (extra cells at grid edges), reductions (combining array values) and regional inputs |
| `physics/dynamics/` | Mass, rheology (how ice deforms under force), drag (resistance to motion), ocean stress (force per area) and the order of ice-motion calculations |
| `physics/transport/` | Advection (movement between cells), cleanup and ridging (ice piling up) |
| `physics/thermodynamics/` | Ice growth and melting, bulk heat fluxes (heat-transfer calculations) and their gradients (rates of change with inputs) |
| `numerics/` | Solvers (methods for solving equations), differentiable building blocks, precise equations and gradient checks |
| `execution/` | Preparing the model, pure timesteps (calculations without outside changes), rollouts (runs through several steps), running and closing resources |
| `io/` | Output saved on a schedule, observations (selected calculated values), storage, loading selected fields and complete restart |
| `setups/` | Preparing supplied setups and command-line examples |

Tests for each part of the model call the module responsible for its calculations
and check the required numerical behavior. Shared `conftest.py` (pytest's common
setup code), standalone process probes (small programs that test behavior),
unchanging `reference_data/`, packaging, benchmark (performance) and runtime
tests remain at the test root. Schema tests check the definitions and single
owners of State, array inputs and OceanGeometry (the ocean geometry definition).
They also check cold imports, loading modules into a fresh Python process.

Activate `.venv-latest` before commands. Install the reproducible CPU (computer
processor) environment with `module load uv/latest` and
`uv pip install -r requirements-test.txt`.
The repository's locked dependency versions target Python 3.14 with JAX (a
library for array calculations) and jaxlib 0.11.2. These match GitHub CI (automated
checks run on GitHub) and the local `.venv-latest`. `requirements-gpu.txt` adds
the matching CUDA 12 plugin, which lets JAX use the local graphics processors
(GPUs). Create the environment with
`uv venv --python 3.14 .venv-latest` and activate it before installation.

Run `pytest tests/ --fast` during development. `VERIS_TEST_SEED` varies the stable
10% sample chosen at collection, when pytest lists tests before running them.
The default is `root`. Run the full suite before commits:

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
workers (separate test-running processes). Tests are grouped by module to share
fixtures (prepared test data and resources) and reuse compilation (translation
into executable calculations). `--partition-cpus` gives each worker a separate
group drawn at regular intervals from the process's permitted CPUs. It does this
before collection or JAX backend initialization, when JAX prepares to use a
processor. Child probes inherit that placement. Disable worker restart so
crashes fail the full run. Use fewer workers if fewer CPUs are available. CPU
CI uses one worker so the four-device sharding probes, which divide arrays
across devices, inherit all four permitted runner CPUs. Splitting them among
three workers can stall XLA collectives (cross-device operations produced by
JAX's compiler) even on JAX 0.11.2.
The full correctness step has no explicit 20-minute timeout.
Dated correctness counts, coverage (how much code tests execute) and elapsed
times live in [CHANGELOG.md](../CHANGELOG.md), alongside the corresponding source
revisions.
Keep solver performance measurements in the separate
[benchmark harness](../benchmarks/README.md).

Each worker starts with a fresh private JAX compilation-cache directory, which
stores compiled calculations for reuse, and shares it with its child probes.
Worker caches are separate because unbounded JAX disk-cache writes are not
locked against simultaneous writes. When the caller has no cache directory or
explicit compile-time policy and JAX retains its default one-second threshold,
the owned session sets the minimum compile time to zero. This lets it reuse
subsecond compilations. Children inherit that policy through environment
variables. Caller-owned cache paths, disabled caches, explicit environment
thresholds, nondefault configured thresholds and the entry-size policy (which
sizes of cached entries to accept) are preserved. Teardown (cleanup after the
run) resets the owned cache, restores configuration and environment settings,
and removes its directory on success or failure. Each full session starts cold,
without a cache from an earlier
session. `test_pytest_runtime.py` checks these resource-management behaviors
against regressions.

Tests run actual compiled JAX functions on small rectangular grids. Choose
precision to match what each assertion must check:

- Run validation, metadata, shape, mask and indexing checks once. These tests
  normally use the default dtype (numerical type); do not repeat them for every
  combination of precisions.
- Use float64, the higher-precision number format, for independent equations
  and precise JVP/VJP/finite-difference references, including gradient
  comparisons across processes. JVP
  (Jacobian-vector product) calculates how outputs respond to an input change
  in a chosen direction. VJP (vector-Jacobian product) calculates input
  sensitivities for a chosen weighted sum of outputs. Finite differences
  estimate rates of change by comparing results from nearby inputs.
- Use float32, the lower-precision number format, for representative operational
  rollouts, restart equivalence and overflow/stability checks, which detect
  numbers outside the representable
  range or calculations that become unstable.
- Test both only for dtype propagation (keeping the required type), conversion,
  precision-sensitive thresholds and the small coupled integration sample,
  which runs ice motion and growth together. Scheduled-output tests retain both
  because they check float64 accumulation (adding samples) with each physics
  dtype and preservation of the global x64 policy (JAX's setting for allowing
  float64 calculations).

Keep dtype assertions explicit in precision tests. A structural assertion used
inside a numerical comparison is still required. For example, check that array
leaves (individual arrays in a nested structure) have equal shapes before
comparing values. This prevents broadcasting, the automatic expansion of
compatible array dimensions, from hiding a failure.

Expected results come from explicit neighborhood indexing (selecting nearby
cells by their indices), scalar threshold rules, momentum balance (balancing
the terms in the ice-motion equations) and finite differences. Nonsmooth tests
cover abrupt switches and thresholds. They check selected AD linearizations
(the local derivative rules chosen by the code) separately from one-sided
slopes (rates of change when approaching from one side).

Whole-package reports include generated version metadata. With user approval,
CI enforces at least 80% coverage of maintained code by excluding only
`veris/_version.py` from the pass/fail check. XML/JSON artifacts (saved report
files) retain whole-package data.
The old geographic setup and its ocean-model dependencies were removed at the
user’s request. CPU CI is enforced. Local GPU requirements are recorded in
`requirements-gpu.txt`; see [CHANGELOG.md](../CHANGELOG.md) for dated hardware
and native GPU verification results. CPU correctness and targeted native GPU
checks have distinct scopes.

Current physical coverage includes periodic transport (movement across connected
opposite grid edges), rheology, wind and ocean stress, EVP momentum limits and
thermodynamic energy/water budgets (accounting for heat and water changes).
EVP is the elastic-viscous-plastic solver for ice velocity and stress. Serial
mode uses `Configuration(use_sharding=False)` when calling functions that use
halos. Fresh-process tests verify real four-CPU-device halo exchange (sharing
boundary values) and reverse-mode sensitivities (derivatives carried back from
outputs to inputs) on 2x2, 1x4 and 4x1 meshes (groups of devices). Separate
reduction probes verify two-process CPU and GPU collectives. These local-machine
checks do not certify networking across cluster computers or MPI communication
launchers.

The [Python lifecycle tutorial](../docs/quickstart/user-guide.rst) provides the
standalone island run. Test harness usage lives here; beginner installation and
first execution live in [Getting started](../docs/introduction/get-started.rst).

Hardware validation uses `requirements-gpu.txt` for the local Tesla P100s.
Require CUDA explicitly while keeping CPU devices available for diagnostic
callbacks, functions called during execution to check or report values:

```bash
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -c 'import jax, pytest; assert jax.default_backend() == "gpu"; raise SystemExit(pytest.main(["tests/", "-q"]))'
```

`grid/test_global_sum.py` launches two real CPU processes. To exercise the same
reduction and adjoint oracles (independent expected sums and reverse-mode
derivatives) on one GPU per process:

```bash
PYTHONPATH=tests:. JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -c 'from reduction_probe import launch_processes; print(launch_processes(platform="cuda"))'
```

Nonsmooth gradient tests distinguish selected JAX linearizations from classical
one-sided slopes. In particular, removing thin ice creates a discontinuous jump.
The derivative is zero on that branch, so it cannot describe what happens when
an input change crosses the removal threshold.

Inside `jax.shard_map`, pass `axis_names=("x", "y")` to `IceVelocities` or
`evp_solver` to obtain global residual diagnostics. These combine scaled changes
in velocity and stress between solver iterations (repeated calculations) across
the whole grid. Callers supply local blocks with two-cell halos. Norms (combined
measures of those changes) exclude those halos before collective reduction
across devices. The default empty tuple preserves serial behavior. The uniform
sharded EVP test checks
only how reductions work; it is not a full distributed coupled run.

The public lifecycle tests cover defining experiments and running, saving and
resuming them: definition and JSON validation, custom ocean geometry/masks,
initialization, pure timesteps, successive host runs (runs managed by ordinary
Python code), forcing position (which input sample a step uses), scheduled means,
snapshots (selected values at one time), selected input and complete serial
restart. Frozen original all-field trajectories check that presets and
numerical kernels (calculation functions)
retain their behavior. Their source history and reproduction instructions are in
[reference_data/README.md](reference_data/README.md).

`execution/test_lifecycle_sharding.py` runs real four-CPU-device experiments in fresh
processes. Its probe covers presets and custom geometry, both boundary modes,
2x2/1x4/4x1 layouts, continued forcing/output and two-cell local extents (physical
region sizes). Separate pure, rollout and closed-wall probes retain JIT
(just-in-time compilation), grad (gradient calculation), JVP, VJP and
finite-difference checks. Serial resource tests run initialization and execution
inside an ambient mesh (a device mesh already active in the caller). They check
single-device placement and restoration of the caller's context. These tests
cannot certify actual multi-host initialization, networking, file ownership or
failure cleanup.

CI also builds documentation with warnings treated as errors, builds fresh
wheel/source archives (installable packages and source bundles), installs the
wheel in a clean target and runs the public examples exactly as written in the
quickstart and README. It runs them outside the source import path, so Python
must use the installed package.
`installed_lifecycle_probe.py` verifies package origin, coupled save/resume,
selected input, custom ocean fields, two-CPU sharding and pure JIT/AD examples.
`test_packaging.py` checks that the source archive includes the complete test
harness, benchmark helpers, reference data, lifecycle design and parallel launcher.

`grid/test_halo_compilation.py` verifies shared halo tracing (JAX's recording of
array operations) with changing values, separate shape/dtype compilations and
independent slice/JVP/VJP expectations.
`io/test_compilation_reuse.py` checks the stable scheduled executor with successive
forcing values and independent output managers. `io/test_callback_lifetime.py`
checks support for arbitrary callable objects, observer overrides and cleanup
of captured resources and objects that own JIT calculations.
`test_parallel_runtime.py` covers CPU placement and private worker-cache
lifecycles, including the required behavior for one-CPU and unsupported-platform
cases. See the verified [reorganization audit](AUDIT.md).
