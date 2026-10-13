# Authoritative Experiment Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every maintained setup and Python/CLI run uses one validated experiment resolution, initialization and execution framework across CPU/GPU serial, mesh and distributed modes.

**Architecture:** Retain Experiment, Configuration, PhysicalConstants, the allocator/ocean metric primitives, regional callbacks, advance/step and Simulation. Separate authored overlays from resolved controls; register data-only setup providers; place mesh/distributed ownership and the public runner in execution. Translate existing drivers and CLI factories into that framework without changing scientific equations or reference startup policies.

**Tech Stack:** Python 3.14.7, JAX/jaxlib 0.11.2, NumPy, Click, existing netCDF/output stack, pytest/xdist, Ruff, ty, Sphinx and Slurm.

**Spec:** [Approved design](../specs/2026-10-13-experiment-framework-design.md).

## Global Constraints

- Work against `jax-only`, never reference `main`; audit base is `9118648ade93de72c696c68c6067fabb5b21391d`.
- Read AGENTS.md/DESIGN.md/CHANGELOG.md; scoped searches only; activate the established `.venv-latest` rather than recreating environments.
- No numerical equation, significant arithmetic-order, frozen archive, tolerance or derivative-convention changes.
- Precedence is shared registry defaults < setup preset/profile < experiment configuration < explicit Python/CLI overrides.
- Grid counts global physical cells; Configuration keeps local cells; storage has two halos per partition edge.
- Preserve base definition and launch-profile grids/steps in the approved design table, including dynamics 600 s/120 EVP, island 600 s/5 EVP and growth daily/5 categories.
- Preserve recursive growth heat, island startup heat outputs, zero inactive dynamics/growth fields and documented island startup mask halos.
- Coupled unequal timesteps remain rejected; no physical subcycling is introduced.
- One pytest session at a time; tests precede production implementation; use `--fast` during development, full correctness/coverage before code commits.
- Maintained coverage gate remains >=80%; no new dependency or alternative numerical/output framework.
- Native local GPUs and local/Slurm parallel runs are authorized; Slurm uses `aegir`. Distinguish local devices, local processes and actual multiple nodes.
- Keep State array-only, resources host-only, and borrowed mesh/distributed services usable after run completion/failure.
- Update CHANGELOG after meaningful units; stage exact files, never all local `test_logs/`.

## Review Focus

1. Repeated scenario/dtype overlays must not promote inherited defaults to explicit controls; pin raw/resolved and precision tests in Task 1.
2. A saved resolved snapshot deliberately pins values while authored export inherits defaults; pin both roundtrip meanings in Task 6.
3. A requested mesh/rank count may be incompatible with global dimensions or available devices; preflight logical requests before effects and test actual topology in Tasks 3/4.
4. A distributed service or mesh may already belong to the caller; preserve ownership and test failed initialization/run/output cleanup in Tasks 4/7.
5. All initialized ice can disappear, or masks/forcing can span partitions; preserve nonsmooth AD conventions and test land/zero-ice/spatial series in Tasks 3/7.

## File ownership and interfaces

| File | Responsibility |
| --- | --- |
| `veris/model/experiment.py` | Immutable authored specification, resolved properties, public overlay API and grouped validation |
| New `veris/model/experiment_resolution.py` | Shared recursive overlay/precision resolution, no devices or files |
| `veris/model/experiment_file.py` | One JSON/array codec for authored/resolved exports and version compatibility |
| `veris/setups/presets.py` | Typed setup descriptors, registration, scientific controls and launch profiles |
| Existing `veris/setups/island.py`, `run_dyn.py`, `run_growth.py` | Field factories and compatibility initializer/CLI translations |
| `veris/execution/experiment_initialization.py` | One provider-driven serial/regional initialization pipeline |
| `veris/execution/initialization.py`, `veris/grid/ocean.py` | Existing allocation and metric primitives; shared relevant validators |
| New `veris/execution/runtime.py` | Distributed launch adaptation, mesh acquisition and resource ownership |
| New `veris/execution/runner.py` | Public host runner over Simulation, preflight and requested outputs |
| `veris/cli/run.py`, `veris/cli/__init__.py` | Source loading, flag translation, inspection/export and reporting |
| Existing `veris/execution/simulation.py`, I/O codec consumers | Read resolved inputs while retaining numerical/resource contracts |
| Existing docs and installed probe | Executable setup blueprint and paired Python/CLI examples |

`Experiment.with_overrides(overrides: Mapping[str, Any]) -> Experiment` merges
authored values and invokes the same validators. Existing settings/constants/
scenario mappings become immutable authored inputs; resolved Configuration and
PhysicalConstants remain existing properties. Add
`resolved_scenario: Mapping[str, Any]` and `resolved_forcing: Forcing` and migrate
internal consumers explicitly. Retain dataclasses.replace behavior for existing
group substitutions, while documenting with_overrides for nested merges.

`SetupPreset` in presets.py supplies name, scenario schema/factory, numerical
defaults, optional grid/steps/profile defaults, physics, field factory, and
startup/forcing policy. `register_preset(preset: SetupPreset) -> None` and
`get_preset(name: str) -> SetupPreset` reject duplicates/unknown names. A factory
signature is `factory(state: State, configuration: Configuration,
constants: PhysicalConstants, scenario: Mapping[str, Any],
global_shape: tuple[int, int], x: np.ndarray, y: np.ndarray,
*, regional: bool) -> Mapping[str, Any]`; State is the shared regional baseline,
not a globally allocated temporary. Setup-specific validation is host-only.

`DistributedExecution` is a validated immutable host request for explicit
coordinator_address/num_processes/process_id/local_device_ids or a supported
cluster_detection_method. `Execution.distributed` optionally holds this request.
The runtime translates existing JAX/Slurm variables only when explicit controls
are absent. Ordinary partitions remain `(1, 1)`; the legacy parallel launch
profile resolves its omitted mesh to all devices before constructing the final
partitioned definition.

`execution.runtime.execution_resources(execution: Execution,
*, mesh: Mesh | None = None) -> context manager[RuntimeResources]` prepares or
borrows resources. RuntimeResources holds actual mesh/backend/topology and
ownership flags; it never enters State. `create_mesh`/`distributed_options` move
here and remain callable by thin legacy adapters.

`veris.run_experiment(experiment: Experiment, *,
overrides: Mapping[str, Any] | None = None, mesh: Mesh | None = None,
output_path: str | Path | None = None,
snapshot_path: str | Path | None = None,
variables: Sequence[str] | None = None) -> RunResult` uses the shared overlay,
runtime and Simulation. Source-file loading remains `cli.run.load_experiment`;
the CLI delegates its resulting Experiment to the public runner.

## Task 1: Correct authored/resolved configuration and override precedence

**Files:** Create `veris/model/experiment_resolution.py` and `tests/model/test_experiment_resolution.py`; modify `experiment.py`, `experiment_file.py`, `presets.py`, existing setup/definition tests and resolved-input consumers in initialization, Simulation and I/O.

**Consumes:** Existing registry constructors, scenario registries and codec merge rules.
**Produces:** `with_overrides`, raw immutable mappings, resolved_scenario/resolved_forcing and reusable host overlay helpers.

- [x] Write `test_scenario_overlay_recomputes_inherited_controls`: island 300 s/7 EVP/25 W m-2 must resolve to 300/7/25; explicit settings 30 s/2 EVP and explicit Qnet=70 still win.
- [x] Write `test_dtype_promotion_recomputes_inherited_defaults`: float32 island/growth promoted to float64 matches direct float64 defaults exactly (34.7 salinity and 1.3 m thickness); authored float32-rounded literals remain exact authored literals.
- [x] Write independent four-layer tables for settings/constants/scenario/forcing/execution/output/steps/grid. Include explicit zero/False/None, series over constant forcing, whole-array replacement, nested mapping merge, unknown keys and incompatible coupled timesteps.
- [x] Run the new file with `pytest tests/model/test_experiment_resolution.py -q`; confirm the stale-control and precision cases fail on current production code before implementation.
- [x] Implement the host overlay/resolution API without device selection. Re-resolve dependent defaults from raw controls; retain defensive ownership/immutability. Migrate consumers to resolved fields without altering physics or startup behavior.
- [x] Run the new file plus existing experiment/preset/initialization/restart tests in one session; then run `pytest tests/ -q --fast`. Update CHANGELOG with RED/GREEN evidence and any corrected behavior assertions.

## Task 2: Put preset data, profiles and future-setup registration behind one contract

**Files:** Modify `presets.py`, `island.py`, `run_dyn.py`, `run_growth.py` and CLI templates; add `tests/setups/test_preset_registry.py`; update metadata/generated documentation consumers.

**Consumes:** Task 1 raw/resolved controls and existing scenario validators.
**Produces:** SetupPreset/get_preset/register_preset and one owner for each setup/profile value.

- [x] Write `test_profiles_preserve_existing_entry_defaults` for every row in the approved defaults table; shared defaults, scientific overrides and launch values must be independently inspectable.
- [x] Write `test_additional_registered_setup_needs_no_core_name_dispatch` using a temporary fourth descriptor, plus duplicate registration, unknown name/profile and invalid scenario cases. Registration must restore test-owned registry state afterward.
- [x] Write `test_growth_shared_values_have_one_authoritative_owner` and island alias-conflict tests. Numerical settings override aliases; nx/ny/steps/dtype cannot enter scientific scenario mappings.
- [x] Run the new tests and confirm missing registry/profile behavior fails.
- [x] Implement descriptors referencing existing authoritative registries. Extract growth's startup field factory; adapt island/dynamics factories to the common signature without changing expressions. Keep serial/regional startup policies as explicit descriptor values.
- [x] Run registry/preset/notebook-equivalence tests and fast mode. Update docs identifying shared/preset/profile ownership and CHANGELOG. Do not add discovery infrastructure or duplicate default literals in CLI decorators.

## Task 3: Use one initializer for presets, geometry and every partition layout

**Files:** Modify `experiment_initialization.py`, setup initializer wrappers, `grid/ocean.py` and applicable shared validators; add `tests/execution/test_provider_initialization.py`; extend initialized/lifecycle sharding probes.

**Consumes:** Task 2 providers and Task 1 validated resolved inputs.
**Produces:** Existing initialize_experiment signature with generic provider dispatch; legacy initializer signatures translated to this API.

- [x] Write frozen all-field step-zero tests for each maintained provider and legacy wrapper, preserving archived startup values and halos.
- [x] Write an adapter invalid-input table for common physical constraints (negative ice, concentration outside [0,1], invalid temperatures/masks/metrics, precision under/overflow), checking meaningful errors before allocation/effects. Respect documented local/halo-inclusive primitive shapes.
- [x] Extend actual CPU sharding tests for all presets/custom geometry on rectangular 1x4, 4x1 and 2x2 layouts, both boundaries, global island indices, fractional/dry masks, partition-crossing fields and spatial forcing. Verify all 70 physical fields against independent serial references.
- [x] Run new cases against current code to identify missing behavior; preserve existing scientific comparisons even if both adapters later share code.
- [x] Replace name-specific serial/sharded branches with a common baseline/provider/input/derived/halo sequence. Shared code owns allocation and global boundary handling. Factory callbacks receive only addressable regions. Translate legacy initializer controls to Experiment, retaining the documented zero-step/initialization-only contract.
- [x] Run provider/adapter/sharding/frozen-reference cases and fast mode; update CHANGELOG and initialization API docs. Run the full CPU/coverage gate before committing the completed configuration/initialization ownership unit.

## Task 4: Centralize execution resources and expose the Python runner

**Files:** Create `runtime.py`, `runner.py`, `tests/execution/test_runtime.py` and `tests/execution/test_runner.py`; modify Execution/codec, `veris/__init__.py`, `run_parallel.py` resource helpers and relevant Simulation/I/O metadata consumers.

**Consumes:** Existing validated Experiment, Simulation and Task 3 initialization.
**Produces:** DistributedExecution, execution_resources/RuntimeResources and public run_experiment returning RunResult.

- [x] Write runner/Python equivalence for all three physics modes and custom geometry with real timesteps, all 70 State fields, Diagnostics and exact completed/time counters.
- [x] Write runtime cold-validation and ownership cases: invalid rank/count/local IDs/coordinator, incomplete environment, explicit versus environment controls, wrong backend/mesh/grid, already initialized service, supplied mesh, creation failure, run failure and snapshot failure. Owned services close once; borrowed services never close. Include distinct device slice IDs: current two-node Slurm job 66298494 fails because jax.make_mesh rejects multi-slice topologies.
- [x] Write real local CPU multi-device and separate-process probes with requested forcing/output. Zero-step runs are valid. A preflight-invalid definition/output must not select devices or create directories.
- [x] Confirm the new runner/runtime tests fail because the interfaces are missing.
- [x] Move the existing all-device mesh and environment adaptation helpers to runtime; implement explicit launch-time precision/backend policy and ownership-aware cleanup. Construct a validated hybrid device mesh when selected devices span slices, rather than calling jax.make_mesh on an unsupported topology. Preflight files and definitions before resources; return RunResult after synchronized execution. Keep imports and inspection free of JAX global-policy changes.
- [x] Run runtime/runner tests, existing lifecycle/resource tests and fast mode. Update public/API docs and CHANGELOG; full CPU/coverage gate before committing the shared runtime unit.

## Task 5: Make central/module CLIs thin adapters with complete overrides

**Files:** Modify central CLI/run loader, three module drivers, templates, io/cli helpers and `run_parallel.slurm`; add `tests/cli/test_framework.py`; extend existing CLI contracts.

**Consumes:** Public runner and with_overrides from Tasks 1/4; launch profiles from Task 2.
**Produces:** Shared execution through every maintained entry; grouped `--set SECTION.KEY=JSON` override parsing.

- [x] Write nonzero Python/central CLI/module-driver equivalence for all presets, custom geometry and both boundary modes, comparing resolved controls and all physical State fields. Copied growth/dynamics templates must execute real steps.
- [x] Write file/preset/CLI precedence cases, omitted flags, zero/False values, nested forcing/output overrides, array replacement and precision promotion. Reject unknown/malformed paths/JSON and conflicting dedicated/generic flags rather than silently choosing an ordering.
- [x] Write zero-step parallel CLI and no-effect invalid preflight tests, preserving no-clobber snapshot/history and variable-selection behavior.
- [x] Run the new file to capture current missing override/adapter behavior.
- [x] Translate flags into one explicit override mapping and invoke the public runner. CLI parser defaults must not overwrite authored config. Consume launch profile owners; remove execution orchestration from setup modules. Retain current installed/module entry points and copied-factory source loading.
- [x] Run all CLI/output/installed-template contracts and fast mode. Update CLI/driver docs and CHANGELOG; full CPU/coverage gate before committing the adapter unit.

## Task 6: Export authored/resolved definitions and publish an executable blueprint

**Files:** Modify experiment codec/API and inspection CLI; add `tests/model/test_experiment_exports.py`, inspection tests and installed blueprint probe; update `docs/cli.rst`, quickstarts, setup gallery/API/package hierarchy and DESIGN.md.

**Consumes:** Task 1 authored/resolved definition, Task 2 profile identity and Task 4 execution controls.
**Produces:** `Experiment.to_dict(*, mode: str = "resolved") -> dict[str, Any]`, modes `resolved`/`authored`; `veris inspect SETUP --export PATH [--authored]` with the same override parser as run.

- [ ] Write tests that resolved exports pin all effective values and authored exports remain sparse. Re-reading authored data and changing scenario/dtype re-evaluates defaults; changing a fully resolved pinned config keeps deliberate literal values. Preserve geometry/forcing immutable array roundtrips and restart's private array hook.
- [ ] Write unknown/malformed mode/schema and no-clobber export tests, source factory/version/profile identity checks and cold inspection assertions.
- [ ] Run the new tests and capture missing modes/inspection behavior.
- [ ] Extend the single codec rather than adding another serializer. Retain version 1 loading; use a new version only for newly represented profile/distributed metadata, with explicit version compatibility tests. Restart remains exact saved State/halos, never initialization plus recomputation.
- [ ] Add the approved paired blueprint Python/CLI example, authored/resolved JSON examples, custom geometry/provider instructions and local/distributed/aegir launches. Execute exact examples from a clean installed wheel with identical resolved controls/physical fields.
- [ ] Run codec/restart/inspection/installed-example checks, strict docs, maintained static checks and fast mode; update CHANGELOG. Include full CPU/coverage verification before any code commit.

## Task 7: Prove scientific, GPU and distributed completion and finish delivery

**Files:** Extend real lifecycle process probes and `tests/execution/test_distributed_experiment.py`; use existing scientific/AD tests; add test-only `tests/slurm_framework_probe.slurm`; update test evidence docs and CHANGELOG.

**Consumes:** Complete shared pipeline from Tasks 1-6.
**Produces:** Requirement-by-requirement evidence, passing maintained gates and reviewed commits on jax-only.

- [ ] Write complete two-process lifecycle comparisons for every physics mode/custom geometry, partition-crossing masks/spatial time forcing, scheduled output writer ownership and exception/shutdown behavior; confirm assertions run on real kernels.
- [ ] Run native single/two-GPU startup/trajectory/AD/output equivalence outside the sandbox, checking platform/device counts before interpreting results. GPU tests must run rather than skip; preserve current tolerance and precision conventions. For tests launching a GPU subprocess from a GPU pytest parent, set XLA_PYTHON_CLIENT_PREALLOCATE=false in the verification environment to prevent competing default memory reservations; retain the initial OOM evidence and do not impose this policy on product imports.
- [ ] Submit a small `aegir` CPU test job with two ranks on one node, then two ranks on two nodes. Require terminal scheduler state and a structured result with all-field numerical/forcing/time/output checks; record job IDs, placement and ExitCode. Use bounded resource requests; inspect a live job rather than restarting after a polling timeout. No GPU request on this CPU partition.
- [ ] Check unchanged reference hashes; run the full coordinated CPU gate and maintained lint/format/annotations/ty. Preserve all equation/stencil/conservation and initial/forcing JIT/grad/JVP/VJP/FD assertions.
- [ ] Build strict Sphinx and fresh wheel/sdist; execute installed literal Python/CLI/mesh/AD examples. Reconcile every approved acceptance row with authoritative current evidence, including hardware scope and any unresolved failure.
- [ ] Request independent whole-change code/test/scientific review; address findings with reproducing tests. Re-run affected checks and a full suite if code changes after the verified gate.
- [ ] Update DESIGN.md/CHANGELOG and exact-file commit contents; commit only passing units on jax-only. Mark the active goal complete only after every requested deliverable is implemented and verified.

## Verification commands

Use one coordinated pytest process, redirect verbose diagnostics to
`test_logs/framework/`, and inspect concise JUnit/aggregate results. Baseline
already passed 139 selected tests (433.029 s); this is prior-source evidence.

```bash
source .venv-latest/bin/activate
JAX_PLATFORMS=cpu pytest tests/ -q --fast
```

Full gate (local host has 48 allowed CPUs; use one worker on CI):

```bash
JAX_PLATFORMS=cpu pytest tests/ -q -n 3 --dist=loadscope --partition-cpus --max-worker-restart=0 --cov=veris --cov-report=term:skip-covered --cov-report=json:test_logs/framework/coverage.json --junitxml=test_logs/framework/full.xml
coverage report --omit=veris/_version.py --fail-under=80
ruff check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py
ruff format --check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py
ruff check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py --select ANN001,ANN002,ANN003,ANN201,ANN202,ANN204,ANN205,ANN206
ty check veris tests benchmarks docs/conf.py setup.py versioneer.pyi --exclude veris/_version.py
sphinx-build -E -W -b html docs test_logs/framework/docs
python setup.py build --build-base test_logs/framework/package-build bdist_wheel --dist-dir test_logs/framework/package-dist sdist --dist-dir test_logs/framework/package-dist
git diff --check
```

When committing a unit, include the tests, code, its relevant docs and CHANGELOG;
stage explicit paths. If a narrow test passes but the full gate fails, resolve
the failure before committing. Do not repeatedly rerun an unchanged full suite
or run a redundant fast sample adjacent to that suite.

## Execution choice and current state

Design and implementation plan are approved; the user selected native execution.
Resolver,
provider and runtime interfaces are tightly connected and pytest must remain
coordinated. Native means root implements all units and a fresh reviewer checks
the full change. Subagent-driven execution gives each task its own implementer
and review cycle. Both retain the specialized scientific/test-quality review
roles and the complete required gates.

Plan self-review: every approved design requirement maps to Tasks 1-7 and the
final acceptance audit; interface names and raw/resolved consumers are consistent;
each review-focus condition has explicit tests. No production/test implementation
has started and no repository commit or remote write has occurred.
