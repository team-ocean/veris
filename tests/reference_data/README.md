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
the barrier. Only `initialize()` is hashed in the island example, because
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

## Original lifecycle trajectories

`lifecycle_legacy_v1.npz` freezes 1,540 arrays from the original setup
initializers and transitions at `fff4cd4e08613e8ec2359e316c94435e81682400`
on `jax-only`, before the shared `advance` implementation. Seven cases cover
island, dynamics and growth with periodic and closed-y boundaries in float64,
plus a float32 island case. Each stores all 70 State fields at steps 0, 1 and 3
and all five Diagnostics fields at steps 1 and 3. Island cooling is prescribed
as 80, 120 and 95 W m-2; growth retains its original recursive heat fields.

`island_unequal_legacy_v1.npz` freezes another 75 arrays after one original
island step with a 600 s dynamics timestep and 3,600 s thermodynamics timestep.
It protects canonical composition of the dynamics and thermodynamics kernels
in `tests/physics/test_unequal_timesteps.py`. Public coupled definitions retain
their equal-timestep requirement.

The JSON manifests record the complete source commit, 49 source-file hashes,
generator/archive hashes, resolved numerical settings/constants, inputs and
Python/JAX/NumPy versions. Tests check archive hashes, shapes and native dtypes
before comparing every field. Float64 comparisons use
`2e-11 * (1 + abs(reference))`; float32 uses `3e-5` with the same scaling.
Reference generation is never part of pytest, and fixtures must not be replaced
with results from the implementation under test.

To reproduce in a separate directory, archive the recorded `jax-only` revision,
copy the two immutable generators there as Python files, and run them outside the source
checkout with `JAX_PLATFORMS=cpu` and `PYTHONPATH` pointing at that archive.
Each generator's command-line arguments are the archived source root, a fresh
output directory and the full recorded commit. Run
`generate_lifecycle_reference.py` first, then
`generate_legacy_unequal_reference.py` using the same output directory. Both
generators reject a source that exposes the new `advance` API. Compare generated
values and manifest provenance with the committed fixtures; do not overwrite
them.

## Notebook initial conditions and dynamics

`standalone_notebooks_v1.npz` freezes 356 float64 arrays from Veris `jax-only`
commit `eb690aa8dc25ed7e6bb3766d658f382eca234b7c` and the example repository's
`jax_halo_exchange` commit `2571998f8feb6a5943b214361c122ca78b44c3a8`.
It contains all 70 current State fields from both original initializers,
and dynamics State at steps 0, 1, 3 and 100 plus ocean stresses. Dynamics reduces
the grid from 1024 to 16 cells per axis and applies the notebook's documented
serial/periodic halo choices;
all 120 adaptive EVP iterations and the original stage sequence are retained.
Growth captures the original one-cell initial column and settings; evolution
uses the maintained shared thermodynamics routines.

The `generate_standalone_notebooks_reference.py.txt` capture source executes
the original initializers and dynamics notebook function with original kernels.
It checks 18 source hashes, rejects escaped imports and existing outputs, and
records both 130-key shared
settings/constants inventories, source commits, environment and array schema
in the manifest. The generator's own bytes are also hashed. Copy it to `.py`,
archive both recorded repositories into separate directories, then run from an
external directory:

```sh
JAX_PLATFORMS=cpu python generate_standalone_notebooks_reference.py \
    MODEL_ROOT EXAMPLE_ROOT FRESH_OUTPUT_ROOT
```

The older lifecycle archive above continues to anchor shared numerical
trajectories. Its tests adapt only known inactive initial field defaults restored
by notebook initialization (zero metrics in growth and zero theta/ATemp in
dynamics). Its frozen bytes, numerical tolerances and all prognostic reference
outputs are retained. Neither archive is regenerated during pytest.

The immutable generator sources use a `.py.txt` suffix to distinguish archived
original capture code from maintained tests. Copy each to its original `.py`
name before execution. Their byte hashes are recorded in the corresponding
manifests.
