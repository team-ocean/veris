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
ruff check tests
ruff format --check tests
ty check tests
```

Tests use actual compiled float64 JAX functions on small rectangular grids.
Expected results come from explicit neighborhood indexing, scalar threshold
rules, momentum balance, and finite differences. Gradients at nonsmooth
thresholds are not covered by the initial mass checks.

Coverage includes generated version metadata. The old geographic setup and
its ocean-model dependencies were removed at the user’s request. The current 70% CI floor prevents loss of the established baseline; it is
not the project target of 80%. Raise the floor as coverage expands. CPU CI is
configured; GPU and distributed execution are still unverified.

Current physical coverage includes periodic transport, rheology, wind and ocean
stress, EVP momentum limits, and thermodynamic energy/water budgets. Serial
mode initializes with `settings["use_sharding"] = False` before importing halo
consumers. A real one-device mesh checks collective halo execution; it does not
verify multiple processors or GPU execution.

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
