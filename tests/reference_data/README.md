# EVP optimization oracle

`evp_pre_barrier.npz` contains float64 CPU output from the source identified in
`evp_pre_barrier.json`, captured before the CPU optimization barrier was inserted.
Arrays store all five EVP fields on a periodic 6×9 interior with two-cell halos;
stress fields are divided by 1000 for comparable numerical tolerances. Six
cases cover fixed/adaptive relaxation, free/no-slip boundaries and 1/4/400 substeps.
The 400-step cases match the iteration depth used by the profiling workload.
Each has open-ocean and single-cell-island values. Open-ocean cases additionally
store a full-field directional JVP and a weighted-output VJP for two parameters:
wind forcing amplitude and ice strength amplitude. Tests independently compare
these derivatives with centered finite differences. Land derivatives are omitted
because the existing adaptive relaxation has a square-root singularity at zero
masked viscosity.

To reproduce, use a separate checkout at the JSON `git_commit`, copy this test
file and the JSON manifest into it, activate `.venv-latest`, then run:

```sh
JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_evp_optimization.py
```

Alternatively, from this tree select the trusted reference package explicitly:

```sh
JAX_PLATFORMS=cpu PYTHONPATH=test_logs/profiling/reference_checkout python tests/test_evp_optimization.py
```

The generator validates the actually imported package, writes this tree’s fixture,
and preserves the verified source commit from the manifest.
The generator checks source hashes before writing and rejects a source containing
the barrier. Only `initialize()` is hashed in the artificial example, because
unrelated whole-step compilation was being developed in the same workspace;
that function was verified identical to the recorded commit. The manifest records
the JAX version used. These regression values preserve the pre-optimization
implementation; the separate equation-based EVP tests remain the physics oracle.
