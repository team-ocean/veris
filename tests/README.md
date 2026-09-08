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
ruff check tests
ruff format --check tests
ty check tests
```

Tests execute actual compiled JAX functions on small rectangular grids,
primarily in float64 with additional float32 stability and communication checks.
Expected results come from explicit neighborhood indexing, scalar threshold
rules, momentum balance, and finite differences. Gradients at nonsmooth
thresholds are not covered by the initial mass checks.

Whole-package reports include generated version metadata. With user approval,
CI enforces at least 80% coverage of maintained code by excluding only
`veris/_version.py` from the gate. XML/JSON artifacts retain whole-package data.
The old geographic setup and its ocean-model dependencies were removed at the
user’s request. CPU CI is configured; GPU hardware remains unverified.

Current physical coverage includes periodic transport, rheology, wind and ocean
stress, EVP momentum limits, and thermodynamic energy/water budgets. Serial
mode initializes with `settings["use_sharding"] = False` before importing halo
consumers. Fresh-process tests verify real four-CPU-device halo exchange and
reverse-mode sensitivities on 2x2, 1x4 and 4x1 meshes. They do not establish
multi-process/MPI or GPU correctness.

Run the standalone artificial-island example in a fresh Python process after
activating `.venv-latest`:

```python
import jax
from veris.setup.artificial import initialize, step

jax.config.update("jax_enable_x64", True)
state, settings = initialize()
for _ in range(3):
    state = step(state, settings, cooling=100.0)
jax.block_until_ready(state)
```

This uses prescribed atmospheric and ocean fields; it demonstrates coupled
dynamics and growth with land masks, rather than an evolving ocean simulation.
Five EVP substeps are for demonstration, not a convergence guarantee.
