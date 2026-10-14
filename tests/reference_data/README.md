# EVP optimization oracle

`evp_pre_barrier.npz` contains float64 output computed on a computer processor
(CPU) from the source identified in `evp_pre_barrier.json`. Float64 is a
higher-precision number format; float32 uses less precision. The data was captured before the CPU
optimization barrier, which limits compiler optimization, was added. EVP means
elastic-viscous-plastic, the ice-motion solver used here. Arrays store all five
EVP fields on a periodic 6×9 interior with two-cell halos. Periodic boundaries
join opposite ends of the grid. Halos are extra cells that hold boundary or
neighboring values. Stress fields are divided by 1000 so they can use comparable
numerical tolerances, the allowed differences between results. Six cases cover
fixed/adaptive relaxation, free/no-slip boundaries and 1/4/400 substeps.
Relaxation controls how repeated solver updates approach a solution. Free/no-slip
boundaries allow or prevent sliding along a wall. Substeps are internal solver
iterations. The 400-step cases use the same number of iterations as the workload
used to measure performance. Each has open-ocean and single-cell-island values.
Open-ocean cases also store a full-field directional JVP and a weighted-output
VJP for two parameters: wind forcing amplitude and ice strength amplitude.
These amplitudes scale the wind input and ice strength. JVP measures field
changes along a chosen input change. VJP measures input sensitivities from a
weighted output. Tests independently compare these derivatives with centered
finite differences, estimates from input changes in both directions. Land
derivatives are omitted because the existing adaptive relaxation has a
square-root singularity at zero masked viscosity: the square root has no finite
derivative there. Viscosity describes resistance to deformation; masking applies
the land/ocean cell values to that quantity.

To reproduce the data, use the unchanged generator from commit ``c39447e`` with
a separate checkout of the numerical source identified by the JSON manifest,
the file that records the capture details. The generator predates the dataclass
API, which uses Python data containers for model objects. The current regression
test uses the new API and should not be copied into a historical source checkout.

```sh
mkdir -p /tmp/veris-evp-capture/reference_data
git show c39447e:tests/test_evp_optimization.py > /tmp/veris-evp-capture/generate.py
cp tests/reference_data/evp_pre_barrier.json /tmp/veris-evp-capture/reference_data/
JAX_PLATFORMS=cpu PYTHONPATH=/path/to/verified/reference python /tmp/veris-evp-capture/generate.py
```

Generated arrays remain under ``/tmp/veris-evp-capture/reference_data`` for
comparison with the committed reference data. These commands do not replace it.

The generator checks source hashes, fingerprints of file contents, before
writing. It rejects a source containing the barrier. Only `initialize()` is
hashed in the island example, because unrelated compilation of whole timesteps
was being developed in the same workspace. That function was checked and found
identical to the recorded commit. The manifest records the JAX version used.
JAX is a Python library for calculations with arrays.
These regression values preserve the implementation before optimization.
The separate EVP tests based on equations remain the reference for the physics.

## Pre-dataclass configuration defaults

`configuration_pre_dataclass.json` records all 131 defaults from
`veris/settings.py` at `c39447e`, including its source SHA256 hash. Configuration
tests compare the new registries, the named tables of model controls, with these
independent historical values. Tests also compare initialized constants and
controls with the exact literal values recorded from their original numerical
modules.

## Original lifecycle trajectories

`lifecycle_legacy_v1.npz` stores 1,540 unchanged arrays from the original setup
initializers and transitions at `fff4cd4e08613e8ec2359e316c94435e81682400`
on `jax-only`, before the shared `advance` implementation. Initializers create
starting fields; transitions advance the model. Seven cases cover island,
dynamics and growth with periodic and closed-y boundaries in float64, plus a
float32 island case. Closed-y boundaries place walls at the ends of the y
direction. Each stores all 70 State fields at steps 0, 1 and 3 and all five
Diagnostics fields at steps 1 and 3. State holds model arrays; Diagnostics holds
step results. Island cooling is prescribed as 80, 120 and 95 W m-2. Growth keeps
its original recursive heat fields, which feed computed heat fluxes into later
steps.

`island_unequal_legacy_v1.npz` stores another 75 unchanged arrays after one original
island step with a 600 s dynamics timestep and 3,600 s thermodynamics timestep.
It checks that the dynamics and thermodynamics kernels, the numerical functions,
are combined in the standard way in `tests/physics/test_unequal_timesteps.py`.
Public coupled definitions still require equal timesteps for ice motion and
freezing/melting.

The JSON manifests record the complete source commit, 49 source-file hashes,
generator/archive hashes, numerical settings/constants after defaults and
overrides are applied, inputs and Python/JAX/NumPy versions. Tests check archive hashes, array
shapes and original numerical data types before comparing every field.
Float64 comparisons use `2e-11 * (1 + abs(reference))`; float32 uses `3e-5` with
the same scaling. Pytest never generates reference data. The reference files must
not be replaced with results from the implementation being tested.

To reproduce the data in a separate directory, archive the recorded `jax-only`
revision and copy the two unchanged generators there as Python files. Run them
outside the source checkout with `JAX_PLATFORMS=cpu` and `PYTHONPATH` pointing
at that archive. Each generator's command-line arguments are the archived source
root, a fresh output directory and the full recorded commit. Run
`generate_lifecycle_reference.py` first, then
`generate_legacy_unequal_reference.py` using the same output directory. Both
generators reject a source that exposes the new `advance` API. Compare generated
values and the source and capture details in their manifests with the committed
reference files. Do not overwrite them.

## Notebook initial conditions and dynamics

`standalone_notebooks_v1.npz` stores 356 unchanged float64 arrays from Veris `jax-only`
commit `eb690aa8dc25ed7e6bb3766d658f382eca234b7c` and the example repository's
`jax_halo_exchange` commit `2571998f8feb6a5943b214361c122ca78b44c3a8`.
It contains all 70 current State fields from both original initializers and
dynamics State at steps 0, 1, 3 and 100 plus ocean stresses, the forces per area
exchanged with the ocean. Dynamics reduces the grid from 1024 to 16 cells per
axis and uses the notebook's documented serial/periodic halo choices.
All 120 adaptive EVP iterations and the original stage sequence are retained.
Growth captures the original one-cell initial column and settings. Its evolution
uses the maintained shared thermodynamics routines for freezing and melting.

The `generate_standalone_notebooks_reference.py.txt` capture source runs the
original initializers and dynamics notebook function with the original kernels.
It checks 18 source hashes and rejects model imports from outside the archived
source and outputs that already exist. Its manifest records both 130-key shared
settings/constants inventories, source commits, environment and array format.
The generator's own bytes are also hashed. Copy it to `.py`, archive both
recorded repositories into separate directories, then run from an external
directory:

```sh
JAX_PLATFORMS=cpu python generate_standalone_notebooks_reference.py \
    MODEL_ROOT EXAMPLE_ROOT FRESH_OUTPUT_ROOT
```

The older lifecycle archive above remains the reference for shared sequences of
model steps. Its tests adjust only known inactive initial field defaults restored
by notebook initialization: zero grid metrics in growth and zero theta/ATemp in
dynamics. Grid metrics describe cell distances and areas. Its saved bytes,
numerical tolerances and all reference outputs for evolving model fields are
retained. Neither archive is regenerated during pytest.

The unchanged generator sources use a `.py.txt` suffix to distinguish archived
original capture code from maintained tests. Copy each to its original `.py`
name before execution. The corresponding manifests record their byte hashes.
