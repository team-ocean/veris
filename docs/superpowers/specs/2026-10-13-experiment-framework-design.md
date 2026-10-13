# Authoritative experiment initialization and execution

Status: approved by the user on 2026-10-13; implementation plan review is next.

## Intent and scope

Consolidate every maintained setup and initialization/run entry point on
`jax-only` into the existing `Experiment`, `initialize_experiment`, `advance`,
`step` and `Simulation` framework. A scientist must be able to define a setup
once, inspect its resolved controls, and run the same definition from Python or
the CLI on serial CPU/GPU, a local device mesh, or distributed processes.

Preserve scientific equations, significant arithmetic order, reference startup
fields, forcing recurrence, boundary conventions and differentiability. Setup
differences become explicit preset data and launch profiles. New setups must
not require name-specific changes to model validation or execution dispatch.

The user authorizes local GPU tests and parallel tests locally or through Slurm
on the `aegir` partition. Actual multi-node results must be distinguished from
local device and local multi-process results.

## Audit of the current implementation

Audit base: `9118648ade93de72c696c68c6067fabb5b21391d`, local `jax-only`.
GitHub confirms that `team-ocean/veris` has the requested `jax-only` branch.
No numerical implementation is taken from `main`.

| Maintained entry | Current responsibility and consolidation needed |
| --- | --- |
| `Experiment`, `builtin`, `from_file`, `to_dict` | Already validate grouped controls and serialize definitions. Preserve authored inputs separately from resolved defaults. |
| `initialize_experiment` | Already handles global dimensions, input validation and regional allocation. Replace separate named serial/sharded setup branches with one provider contract. |
| `execution.initialization.initialize` | Canonical low-level allocation primitive with local, halo-inclusive storage semantics. Keep this documented primitive; do not present it as a second experiment API. |
| `grid.ocean.initialize_from_ocean` | Canonical halo-inclusive metric adapter. Reuse its metric equations; align common mask/range validation with the experiment path. |
| `island.initialize`, `run_dyn.initialize`, `run_growth.initialize` | Currently resolve controls and allocate independently. Retain public signatures as thin translations to the shared experiment initializer. |
| `Simulation.initialize`, `run`, `step`, restart and I/O | Already own numerical lifecycle and clock. Preserve pure numerical carry, output continuation and borrowed mesh semantics. |
| `advance`, numerical `step` | Already centralize physical ordering and pure rollout. Preserve these interfaces and their AD contracts. |
| Central CLI and three copied `setup.py` factories | Already load Experiment definitions. Move overrides and execution into public Python functions; add full grouped overrides and configuration inspection/export. |
| Dynamics/growth/parallel module CLIs | Duplicate definition construction, output handling and launch policy. Translate flags to the same public runner, retaining explicit launch profiles. |
| `run_parallel.slurm` | Keep as a launch example invoking shared execution with explicit partition resources. |
| Documentation examples and benchmark runner | Migrate user examples to the shared API. The benchmark remains a separate measurement harness using canonical allocation/execution primitives. |

There is no maintained `examples/` directory or geographic Veros setup in the
current checkout. The prescribed-ocean example is maintained in
`docs/quickstart/custom-ocean.rst`. Frozen reference generators remain provenance
tools, rather than setup entry points to refactor.

### Confirmed correctness defects

`Experiment.__post_init__` stores resolved preset settings, normalized scenario
defaults and injected island forcing in the fields used as constructor inputs.
Later `dataclasses.replace` therefore promotes inherited values into explicit
overrides. A read-only host probe reproduced:

```python
base = Experiment.builtin("island")
changed = replace(base, scenario={
    **base.scenario,
    "islandTimeStep": 300,
    "islandEVPsteps": 7,
    "islandCooling": 25,
})
```

The scenario reports `300/7/25`, while effective timestep, EVP iterations and
cooling remain `600/5/100`. A second host probe found float32-to-float64 promotion
retains rounded inherited defaults: island salinity becomes
`34.70000076293945` rather than `34.7`, and growth ice thickness becomes
`1.2999999523162842` rather than `1.3`.

Shared initialization branches explicitly on `island`, `dynamics` and `growth`
twice. It calls full setup initializers in serial mode and reconstructs their
fields differently for sharding. The central CLI imports execution resource
helpers from the parallel dynamics example, which independently repeats their
lifecycle. These are architectural gaps even where current tests pass.

### Intentional defaults to retain explicitly

| Context | Grid | Steps | Numerical/field policy |
| --- | --- | ---: | --- |
| Base island definition | 8 x 12 | 0 | 600 s, five EVP iterations, prescribed cooling |
| Base dynamics definition | 8 x 12 | 0 | 600 s, 120 adaptive EVP iterations, fixed wind snapshot |
| Base growth definition | 2 x 2 | 0 | Daily timestep, five categories, recursive heat flux |
| Island copied example | 8 x 12 | 3 | Island scientific preset |
| Dynamics copied example | 16 x 24 | 100 | Dynamics scientific preset |
| Dynamics standalone driver | 1024 x 1024 | 100 | Dynamics scientific preset |
| Parallel dynamics driver | 1024 x 1024 | 1000 | Dynamics preset; nine-field output; automatic all-device mesh when omitted |
| Growth copied example/driver | 2 x 2 | 150 | Growth scientific preset |

Keep these public launch behaviors through named profile overlays or explicit
example configurations. Numerical/scenario literals have one registry owner;
Click declarations and templates consume those owners. Profile selection belongs
to the preset layer, rather than introducing another precedence level.

## Approaches considered

1. Extend the existing immutable definition and host facade with a resolver,
   data-provider contract and shared resource runner. Recommended: preserves
   proven numerical/output primitives and makes ownership explicit.
2. Keep all initializers and introduce a dispatcher around them. This leaves
   validation, default precedence and serial/sharded initialization with multiple
   owners, so it does not meet the requested final state.
3. Introduce a replacement experiment engine. This duplicates existing
   lifecycle/codec/halo machinery and increases scientific regression risk
   without providing an additional required capability.

## Definition, ownership and precedence

`Experiment` remains the public immutable specification. Its existing groups
continue to cover global grid, geometry and ocean masks, initial ice fields,
prescribed forcing, numerical controls, constants, timesteps, physics mode,
execution and output. Add only the controls needed for explicit setup profiles
and distributed launch configuration; numerical State never holds host resources.

Constructor mappings retain immutable, sparse, authored values without
dtype-normalizing them or inserting inherited defaults. Store validated resolved
scenario, forcing, Configuration and PhysicalConstants separately. Internal
initialization and Simulation use the resolved values. Existing public consumers
that read resolved controls must be migrated explicitly and documented.

One shared resolver applies:

```text
shared registry defaults
    < selected setup preset and launch profile
    < experiment configuration
    < explicit Python or CLI overrides
```

Resolve dependent values after overlay merging, then normalize to the requested
precision. Re-resolving an unchanged definition is idempotent. Changing dtype
re-evaluates raw inherited defaults; deliberately authored rounded scalars remain
deliberately rounded. Changing a scenario updates its inherited numerical or
forcing choices, while explicit settings/forcing continue to win.

Add `Experiment.with_overrides(mapping)` for the common immutable overlay path.
Python and CLI use the same grouped keys and validators. Merge nested mappings
by key; replace arrays, sequences and scalar values as wholes. Omitted options
are absent from the override layer. Explicit `0`, `False` and allowed `None`
values are never treated as omitted. Unknown groups/keys and incompatible
options fail with group/field context before device selection or file creation.

Owners remain:

| Controls | Authoritative owner |
| --- | --- |
| Universal numerical inputs and dependent formulas | `SETTINGS` and `Configuration` |
| Physical inputs and dependent formulas | `PHYSICALCONSTANTS` and `PhysicalConstants` |
| Allocated field metadata/defaults | `VARIABLES` |
| Output/calendar/stream defaults | Existing I/O registries and validators |
| Scientific scenario data and intentional numerical overrides | Each setup's registered descriptor and scenario schema |
| Global dimensions and local partition conversion | `Grid`, shared resolution and regional initialization |
| Mesh/distributed launch validation and ownership | Shared execution runtime |

Keep island timestep/EVP aliases as an explicit compatibility translation to
numerical settings, with explicit numerical settings taking precedence. Growth
grid/run controls become profile data, rather than competing scenario owners.
Where growth values equal shared registry defaults, reference that owner instead
of repeating the literal.

## Setup providers and initialization

Extend `veris.setups.presets` with a small typed setup descriptor/registry:
scientific defaults, scenario validation, supported physics, startup field
factory, and declarative startup/forcing policies. Reject duplicate registrations.
Do not introduce plugin discovery, a dependency container or a second schema
framework. A custom `setup.py` can still return an ordinary Experiment with
OceanGeometry and prescribed fields without registration.

A field factory receives validated controls, constants, global shape and global
cell indices for the requested region. It supplies fields/data; it does not
construct Configuration, allocate a full model, select devices, create a mesh,
initialize distributed JAX, fill global halos, advance physics or open files.
The shared initializer allocates the baseline, invokes the same provider for
serial storage or addressable regions, applies explicit initial/forcing inputs,
updates the appropriate derived fields and establishes boundary halos.

Reuse `regional_state`, `regional_geometry`, `regional_forcing`, the existing
ocean metric adapter and field-specific halo helpers. Global physical dimensions
remain distinct from local Configuration dimensions and packed storage. Spatial
forcing partitions x/y only, retaining the whole time axis. Each rank allocates
only its addressable numerical regions. Global land features and forcing retain
global indices; internal y boundaries never become closed basin walls.

Reference startup policies are explicit descriptor data. Preserve zero unused
dynamics/growth fields, island serial outermost startup mask halos, island zero
startup heat outputs with cooling applied before stepping, and growth recursive
Qnet/Qsw. Do not globally recompute legacy intermediates or refresh every startup
halo merely to make providers uniform. Generic custom geometry applies its
existing masked-ice/velocity and derived-load/inverse-thickness policy.

Perform common host validation before allocation: geometry shape/metric/mask
constraints, precision range and reciprocals, initial/forcing physical bounds,
derived-range limits, grid partitioning, physics/timestep compatibility and
forcing length. Low-level adapters retain their documented local/storage shape
contracts while sharing these validators wherever the inputs have the same
scientific meaning. A zero-step run is valid through every run adapter.

## Shared execution and resource lifecycle

Move mesh/distributed helpers into `veris.execution.runtime`. Add a public
`run_experiment(experiment, *, overrides=None, mesh=None, output_path=None,
snapshot_path=None, variables=None)` runner returning the existing RunResult.
It resolves and preflights definitions and output paths, acquires requested
runtime resources, uses Simulation, synchronizes results, writes requested
output and closes owned resources on normal or exceptional exits.

`Simulation.initialize` continues to support expert callers supplying an
already prepared mesh and distributed service. Those resources remain borrowed.
The runner must not shut down a caller-owned distributed service or mutate JAX
policy as an import side effect. Runtime precision/backend setup is explicit
at launch and precedes device enumeration; cold definition inspection allocates
no devices. Reuse current all-device mesh semantics unless an explicit device
selection is introduced and tested as part of this contract.

Explicit distributed controls validate coordinator/count/rank/local IDs or a
supported cluster detection method. Existing JAX/Slurm environment variables
are adapted once by this runtime, with explicit specification values taking
precedence. Distributed launches require an explicit backend. Record the actual
backend, process topology and resolved partitions after resource selection.
An omitted parallel-driver mesh retains its all-device behavior through its
launch profile; an ordinary Execution defaults to serial.

Central CLI run, module drivers and Python calls invoke this runner. CLI modules
own only parsing, source loading and human-readable reporting. Convenience flags
translate to grouped overrides. Add a grouped `--set SECTION.KEY=JSON` option for
controls without dedicated flags; malformed JSON/paths, unknown keys and
conflicting aliases produce a concise error. Defaults come from definitions or
profiles, rather than parser literals that overwrite file values.

Keep the existing numerical `advance`/`step` and Simulation clock/forcing/output
carry unchanged in meaning. All three physics modes use their established
ordered kernels. Coupled unequal timesteps continue to fail until a separately
specified physical subcycling scheme exists. No new numerical model is added.

## Inspection, export and setup blueprint

Reuse the authoritative experiment codec and its typed array hook. Preserve
reading version 1 definitions/restarts. A resolved snapshot pins all written
values; reading it treats those values as experiment configuration. A reusable
authored export retains sparse inputs and inherits future preset defaults.
Expose both modes clearly through `Experiment.to_dict` and configuration
inspection/export commands; never infer provenance from equality to a default.
If added execution/profile fields require a new schema version, implement that
version in the same codec with explicit version 1 compatibility and tests.

Resolved export includes global dimensions, all effective numerical/physical
inputs, scenario and forcing policy, geometry/custom fields, output controls and
setup/profile identity. Actual runtime backend/topology is execution metadata,
not a guessed value for `backend=None`. Setup-generated fields retain their
factory identity and complete resolved controls; explicitly prescribed data is
serialized. Preserve restart's exact evolved State/halo restoration and do not
silently reinitialize it through setup factories.

Add a concise setup blueprint to the existing documentation: provide geometry
and physical masks or a registered field factory; supply initial/forcing fields;
select physics and timesteps; return Experiment. The primary paired example is:

```python
from pathlib import Path
from veris import run_experiment
from setup import build_experiment

result = run_experiment(
    build_experiment(),
    overrides={"steps": 3, "execution": {"backend": "cpu"}},
    snapshot_path=Path("python.nc"),
)
```

```bash
veris run setup.py --steps 3 --backend cpu --output cli.nc
```

Both resolve the same scientific controls and produce equivalent physical fields.
Document JSON loading, authored/resolved export, local CPU/GPU meshes, distributed
Python and `aegir` launches alongside the example. Execute the literal examples
from a clean installed wheel.

## Verification and acceptance

Write regression tests before each implementation unit. Root coordinates one
pytest session at a time; independent reviewers do not start additional suites.
Use deterministic fast mode during development and the full suite before a code
commit, avoiding a redundant adjacent fast run.

| Requirement | Evidence required before completion |
| --- | --- |
| One specification and one owner per default | Source ownership inventory, descriptor/default tests and rejection of duplicate/unknown controls |
| Four-layer precedence | Independent expected-value tables across every preset/group, omitted/zero/False values, whole-array replacement, scenario re-resolution and dtype promotion |
| Generic future setups | A test-only additional provider and a custom geometry setup initialize/run without editing shared name dispatch |
| Every maintained adapter | Nonzero Python/central CLI/module-driver trajectories for every preset and custom geometry, with identical resolved controls |
| Numerical preservation | Unchanged frozen archive hashes and tolerances; all-field startup/trajectory and Diagnostics comparisons |
| Differentiability | Existing equation/conservation and finite-difference/JIT/grad/JVP/VJP contracts, plus provider/refactor regression checks |
| Local partition correctness | Rectangular 1x4, 4x1, 2x2 CPU layouts for presets/custom geometry, both boundaries, masks across partitions and spatial/time forcing |
| GPU support | Native single/multiple-GPU initialization, trajectories, scheduled output and serial/parallel equivalence |
| Distributed support | Real separate-process full lifecycle, writer ownership and shutdown/failure tests; actual multi-node `aegir` run where resources are available |
| Resource lifecycle | Owned services close once on success/failure; borrowed resources stay usable; preflight failures have no file/device effects |
| Inspection and blueprint | Codec authored/resolved roundtrips, metadata assertions and clean-installed literal Python/CLI examples |
| Required quality gates | Maintained Ruff/format/annotations/ty, full CPU correctness/coverage >=80%, strict Sphinx, clean wheel/sdist and installed probes |
| Documentation and delivery | DESIGN/API/setup/CLI references and CHANGELOG updated; small passing commits on `jax-only` |

Keep the independent scientific protections: lifecycle archive with 1,540 arrays
across seven cases; unequal-timestep kernel-composition archive with 75 arrays;
notebook initialization/100-step dynamics archive with 356 arrays; equation,
stencil, conservation and finite-difference tests. Python/CLI equivalence must
supplement these oracles, never replace them.

Prior verification recorded in CHANGELOG is 1,382 CPU passes/four GPU skips,
all four subsequently passed on native GPUs, and maintained coverage 94.38%.
Those results establish the previous baseline; they do not verify this proposal.
Fresh audit results and later implementation results are recorded separately.

## Work sequence and review boundary

After written-spec approval, produce the implementation plan with test-first
units: resolver/provenance; provider/default ownership; shared initialization;
shared runtime/runner; all CLI adapters; configuration export/blueprint; complete
scientific/hardware/packaging verification and documentation. Review the plan
and select its execution method before implementation, as required by the
explicitly invoked Superpowers workflow.

This proposal is not a completion claim or a reduction of the requested scope.
Implementation awaits written-spec and plan review in the explicitly invoked
workflow. Repository commits will obey AGENTS.md's full-suite gate rather than
committing an unverified design in isolation.
