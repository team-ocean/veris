# Scientific package hierarchy

The goal is a responsibility-based source tree, preserved numerical behavior and
public APIs, and a full correctness suite taking at most 1200 seconds. Work stays
on `jax-only`. Current maintained coverage is 4058/4296 = 94.46%; preserve those
executed scientific lines, all frozen references, and existing numerical tolerances.

## Chosen approach

Move authoritative implementations into shallow domain packages; retain explicit,
statically typed compatibility modules at previous paths. Both paths resolve to the
same module object, so private helpers, monkeypatches, function signatures and JAX
PyTree identities stay consistent. Newly serialized class names identify canonical
owners; old pickle imports remain resolvable. Root `veris.__all__` stays unchanged.

Keeping implementations flat with only domain facades would leave ownership
unclear. Removing old modules would break clients. Separate implementations plus
wrappers that copy attributes would break patches of function globals. Use module
aliases with explicit re-exports for static type checking.

## Tree

```text
veris/
  model/           configuration, constants, metadata, array/state contracts,
                   diagnostics, experiment definitions, forcing, definition codec
  grid/            geometry input schema, ocean adapter, halos, averaging,
                   reductions, addressable regions
  physics/
    dynamics/      ordered dynamics/transport stage, mass, rheology, ocean stress
    transport/     advection, cleanup and ridging
    thermodynamics/ growth, CESM and MITgcm heat fluxes
  numerics/        differentiable primitives, momentum dispatch, EVP,
                   free drift, surface-temperature iteration
  execution/       allocation, experiment initialization, pure advance,
                   rollout, output rollout and host Simulation
  io/              existing calendar/storage/output/scan, lifecycle I/O,
                   complete restart and restart array codec
  setups/          maintained island/dynamics/growth/parallel entry points, presets
  <legacy>.py      typed compatibility aliases
  _version.py      generated version metadata (unchanged)
```

Small package initializers document responsibilities and do not import upper
layers. `model.arrays` owns shared input aliases; `model.types` owns State, Parameter
and kernel contracts. `grid.geometry` owns OceanGeometry and imports only array
aliases; `model.types` retains its compatibility re-export. Geometry allocation
remains separate from this cold schema. Move shared packed-halo removal out of
the parallel CLI into `grid.halos`; the CLI retains its existing callable export.

## Module migration

| Previous module | Canonical module |
| --- | --- |
| `veris._typing` | `veris.model.types` |
| `veris._metadata` | `veris.model.metadata` |
| `veris.configuration` | `veris.model.configuration` |
| `veris.physical_constants` | `veris.model.physical_constants` |
| `veris.variables` | `veris.model.variables` |
| `veris.diagnostics` | `veris.model.diagnostics` |
| `veris.experiment` | `veris.model.experiment` |
| `veris.experiment_file` | `veris.model.experiment_file` |
| `veris.forcing` | `veris.model.forcing` |
| `veris.fill_overlap` | `veris.grid.halos` |
| `veris.averaging` | `veris.grid.averaging` |
| `veris.global_sum` | `veris.grid.reductions` |
| `veris.model_regions` | `veris.grid.regions` |
| `veris.setups.ocean` | `veris.grid.ocean` |
| `veris.dynamics` | `veris.physics.dynamics.driver` |
| `veris.dynamics_routines` | `veris.physics.dynamics.rheology` |
| `veris.area_mass` | `veris.physics.dynamics.area_mass` |
| `veris.ocean_stress` | `veris.physics.dynamics.ocean_stress` |
| `veris.advection` | `veris.physics.transport.advection` |
| `veris.clean_up` | `veris.physics.transport.cleanup` |
| `veris.growth` | `veris.physics.thermodynamics.growth` |
| `veris.heat_flux_CESM` | `veris.physics.thermodynamics.heat_flux_cesm` |
| `veris.heat_flux_MITgcm` | `veris.physics.thermodynamics.heat_flux_mitgcm` |
| `veris._ad` | `veris.numerics.autodiff` |
| `veris.dynsolver` | `veris.numerics.momentum` |
| `veris.evp_solver` | `veris.numerics.evp` |
| `veris.freedrift_solver` | `veris.numerics.freedrift` |
| `veris.solve4temp` | `veris.numerics.surface_temperature` |
| `veris.initialization` | `veris.execution.initialization` |
| `veris.model_initialization` | `veris.execution.experiment_initialization` |
| `veris.model_physics` | `veris.execution.advance` |
| `veris.integration` | `veris.execution.rollout` |
| `veris.integration_output` | `veris.execution.rollout_output` |
| `veris.simulation` | `veris.execution.simulation` |
| `veris.simulation_io` | `veris.io.lifecycle` |
| `veris.restart` | `veris.io.restart` |
| `veris.restart_arrays` | `veris.io.restart_arrays` |
| `veris.presets` | `veris.setups.presets` |
| `veris._typing.ArrayInput`, `MaskInput` | `veris.model.arrays` |
| `veris._typing.OceanGeometry` | `veris.grid.geometry` |
| `veris.setups.run_parallel.remove_halos` | `veris.grid.halos.remove_halos` |

`veris.settings` continues to re-export Configuration/SETTINGS/Parameter. Setup
module/CLI names and the top-level Experiment, Simulation, advance and step APIs
remain available. New code and active documentation use canonical imports.
Historical plans and scientific capture artifacts retain original paths.

## Dependency policy

Schemas and numerical settings do not depend on execution. Numerical kernels
consume State/settings/constants, grid helpers and shared differentiable primitives.
Execution composes physics and owns device/resource/time orchestration. I/O stores
and samples data without depending on setup CLI implementation. Setup drivers
remain compatibility entry points to execution. Keep codec/preset convenience
imports deferred where host methods need upper layers; no import-time cycles.
Do not split scientific equations or change arithmetic order during file moves.
No maintained module exceeds 800 lines.

## Tests and runtime

Write migration contracts before moves: all module identities, exported symbols,
patched globals, cold import orders, unchanged root exports, old pickle resolution,
State tree leaves, real compiled numerical references and canonical type checking.
Move existing test files into matching responsibility directories, preserving their
assertions, reference data, collection and standalone child probes. Keep explicit
compatibility coverage as well as canonical-path coverage.

Measure runtime from the same environment with one pytest instance. Existing full
CPU run is 2219.44 s; cache threshold and child AD staging are investigation targets,
not established savings. Retain all scientific assertions, dtypes where they are the
contract, gradients, sharding layouts and output/restart checks. Use private cold
session caching and shared immutable reference preparation only after evidence.
Performance measurements remain in the separate benchmark harness. Do not meet
the budget by using --fast, skipping tests, loosening tolerances or warm prior caches.

## Completion evidence

Full CPU correctness must pass in <=1200 seconds with maintained coverage near the
baseline and no missing scientific coverage. Verify both local GPU-specific cases
with explicit CUDA selection. Run maintained Ruff lint/annotation checks, formatting,
ty, dependency validation, strict Sphinx, wheel/sdist build and installed public
examples. Verify all canonical modules and compatibility files are in the wheel;
the source archive includes complete tests, helpers, references, docs and benchmarks.
Update DESIGN.md, migration/API docs, tests/AUDIT.md and CHANGELOG.md. Commit only
after the full correctness suite passes. No unrelated features or reference changes.
