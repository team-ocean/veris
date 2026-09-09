# Numerical tests

Activate `.venv-latest` before commands. Install the reproducible initial CPU environment
with `module load uv/latest` and `uv pip install -r requirements-test.txt`.
The lock targets Python 3.14 with JAX 0.11.1, the latest stable releases
available during setup on 2026-09-08. Create the environment with
`uv venv --python 3.14 .venv-latest` and activate it before installation.

Run `pytest tests/ --fast` during development. `VERIS_TEST_SEED` varies the stable
10% collection sample; the default is `root`. Run the full suite before commits:

```sh
pytest tests/ --cov=veris --cov-report=term:skip-covered
coverage report --omit=veris/_version.py --fail-under=80
ruff check veris tests doc/conf.py setup.py --exclude veris/_version.py --ignore N999
ruff format --check veris tests doc/conf.py setup.py --exclude veris/_version.py
ty check veris tests doc/conf.py setup.py --exclude veris/_version.py
```

Tests execute actual compiled JAX functions on small rectangular grids,
primarily in float64 with additional float32 stability and communication checks.
Expected results come from explicit neighborhood indexing, scalar threshold
rules, momentum balance, and finite differences. Nonsmooth tests check selected
AD linearizations and one-sided slopes separately.

Whole-package reports include generated version metadata. With user approval,
CI enforces at least 80% coverage of maintained code by excluding only
`veris/_version.py` from the gate. XML/JSON artifacts retain whole-package data.
The old geographic setup and its ocean-model dependencies were removed at the
user’s request. CPU CI is enforced; local hardware validation uses two Tesla
P100 GPUs with driver 580.173.02 and JAX 0.11.1 CUDA 12.

Current physical coverage includes periodic transport, rheology, wind and ocean
stress, EVP momentum limits, and thermodynamic energy/water budgets. Serial
mode uses `Settings(use_sharding=False)` when calling halo
consumers. Fresh-process tests verify real four-CPU-device halo exchange and
reverse-mode sensitivities on 2x2, 1x4 and 4x1 meshes. Separate reduction probes
verify two-process CPU and GPU collectives. These local-machine checks do not
certify multi-node networking or MPI launchers.

Run the standalone artificial-island example in a fresh Python process after
activating `.venv-latest`:

```python
import jax
from veris.setup.artificial import initialize, step

jax.config.update("jax_enable_x64", True)
state, settings, constants = initialize()
for _ in range(3):
    state = step(state, settings, constants, cooling=100.0)
jax.block_until_ready(state)
```

This uses prescribed atmospheric and ocean fields; it demonstrates coupled
dynamics and growth with land masks, rather than an evolving ocean simulation.
Five EVP substeps are for demonstration, not a convergence guarantee.

Hardware validation uses `requirements-gpu.txt` for the local Tesla P100s.
Require CUDA explicitly while retaining CPU devices for diagnostic callbacks:

```bash
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false python -c 'import jax, pytest; assert jax.default_backend() == "gpu"; raise SystemExit(pytest.main(["tests/", "-q"]))'
```

`test_global_sum.py` launches two real CPU processes. To exercise the same
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
