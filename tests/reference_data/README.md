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

To reproduce, use the immutable generator from commit ``c39447e`` with a
separate checkout of the numerical source identified by the JSON manifest.
The generator predates the dataclass API; the current regression test uses the
new API and should not be copied into a historical source checkout.

```sh
mkdir -p /tmp/veris-evp-capture/reference_data
git show c39447e:tests/test_evp_optimization.py > /tmp/veris-evp-capture/generate.py
cp tests/reference_data/evp_pre_barrier.json /tmp/veris-evp-capture/reference_data/
JAX_PLATFORMS=cpu PYTHONPATH=/path/to/verified/reference python /tmp/veris-evp-capture/generate.py
```

Generated arrays remain under ``/tmp/veris-evp-capture/reference_data`` for
comparison with the committed fixture; these commands do not replace it.

The generator checks source hashes before writing and rejects a source containing
the barrier. Only `initialize()` is hashed in the artificial example, because
unrelated whole-step compilation was being developed in the same workspace;
that function was verified identical to the recorded commit. The manifest records
the JAX version used. These regression values preserve the pre-optimization
implementation; the separate equation-based EVP tests remain the physics oracle.

## Pre-dataclass configuration defaults

`configuration_pre_dataclass.json` records all 131 defaults from
`veris/settings.py` at `c39447e`, including its source SHA256. Configuration
tests compare the new registries with these independent historical values.
Additional initialized constants and controls are tested against the exact
literals inventoried from their original numerical modules.
