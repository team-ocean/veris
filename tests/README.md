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

Coverage currently includes generated version metadata and the legacy Veros
setup. The current 52% CI floor prevents loss of the established baseline; it is
not the project target of 80%. Raise the floor as coverage expands. CPU CI is
configured; GPU and distributed execution are still unverified.

Current physical coverage includes periodic transport, rheology, wind and ocean
stress, EVP momentum limits, and thermodynamic energy/water budgets. Serial
mode initializes with `settings["use_sharding"] = False` before importing halo
consumers. A real one-device mesh checks collective halo execution; it does not
verify multiple processors or GPU execution.
