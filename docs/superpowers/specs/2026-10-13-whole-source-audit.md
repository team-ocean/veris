# Whole-source compatibility audit

Continue the user-authorized canonical cleanup after the experiment framework.
Work and commit on `jax-only` with native implementation and specialized audit,
documentation and final review assistance. User authorization to remove obsolete
APIs takes precedence over the framework's earlier adapter-retention requirement.

## Source coverage

Read every maintained Python module: package facade; all model, grid, physics,
numerics, execution, I/O, setup and CLI modules, including three setup templates.
Reviewed benchmark, packaging, CI, SLURM launch and public examples. The complete
76-module Python package is listed in the verified package inventory under
`test_logs/compatibility-audit/narrow-package-inventory.json`. Generated
Versioneer source stays generated; immutable historical capture generators,
metadata and archives stay scientific provenance. Source audit covers 37 grid,
physics, numerics, execution and benchmark files; API audit covers model, I/O,
CLI and root. Root reviews all setup files and integration callers.

## Remove

- Duplicate growth scenario `nx`, `ny` and `steps` inputs. Grid defaults and
  launch durations belong to the preset provider/profiles; scenario metadata
  contains only physical column controls.
- Three setup `initialize` translation APIs; `_initialize_adapter` and
  `_UnknownScenarioError`; `_InitializationExperiment` with initialization-only
  clock and singleton-mesh exceptions. Use `Experiment.builtin` with global
  `Grid`, grouped controls and `initialize_experiment` everywhere. Low-level
  unequal-clock equations use immutable Configuration updates directly.
- `run_parallel.create_mesh`/`distributed_options` reexports and `parse_args`
  default-insertion adapter. Runtime owns mesh/bootstrap; Click parses explicit
  options and Experiment profiles own defaults.
- `io.cli.output_settings`, duplicating `driver_definition` output translation,
  and uncalled `prepare_output_paths`, duplicating runner path preparation.
- `make_sharded_fill_overlap`, duplicating the canonical halo API's shard-map
  binding. Tests call `fill_overlap` inside a host mesh context, including AD.
- Module-driver nonfinite exception translations to RuntimeError or
  FloatingPointError. Preserve the runner's ValueError and concise console exit.

## Retain with evidence of value

- OceanGeometry conversion has a real halo-inclusive metric/mask validation
  contract, distinct from runnable global physical inputs. General allocation
  supplies raw State fields for numerical kernels and complete metadata tests.
- `advance`, `step`, timed/scheduled rollout, Simulation, run_experiment and
  execution resources own physics, AD iteration, timings, calendar reduction,
  persistent lifecycle, delivery and resource ownership respectively.
- Directional flux APIs, CESM/MITgcm flux variants, constitutive stress,
  numerical AD helpers, solver dispatch and CUDA optimization barriers implement
  distinct scientific or measured execution contracts. Field-aware halos and
  regional callbacks implement real boundary and distributed storage semantics.
- Tracing guards, weak callback helpers, immutable authored/resolved controls,
  typed restart arrays and calendar aliases enforce current AD, lifetime,
  validation, storage and interoperable calendar contracts.
- `explicitDrag` and unused historical physical metadata preserve the complete
  version-one checkpoint schema and independent registry provenance. No new
  compatibility path is added to hide their removal.
- Vendored Versioneer and generated version metadata retain Git-tag, wheel and
  sdist release-version contracts; they are packaging machinery, not maintained
  numerical or legacy initialization APIs.
- Curated root/io exports, module CLIs, provider factories, selected snapshots,
  complete restart and direct host/device output paths remain current features.

## Acceptance

Preserve every numerical kernel, frozen reference byte, scientific tolerance,
prescribed-ocean example and gradient check. Adapt obsolete API tests explicitly;
replace adapter self-comparisons with physical-value assertions. Verify canonical
imports, all-field independent primals, JIT/grad/JVP/VJP, real sharding, CLI/files,
strict docs and clean-installed examples. Run Ruff lint/format/annotations, ty,
shell syntax, full CPU correctness and maintained coverage >=80%, then native
GPU cases. Independent review and all gates precede the git commit. Keep full
correctness within the existing 1200-second project target; record failed runs.

## Final hardware correctness extension

The requested final two-node aegir run exposed intermittent partition-boundary
errors despite identical settings and compiled programs. Repeated all-field
trajectories reproduce the failure; an ordering barrier before the final halo
refresh removes it in the diagnostic control. Establish that ordering in the
execution layer after a maintained native regression reproduces the issue.
The barrier applies only to sharded multi-process execution. Applying it to local
execution exposed an in-process collective AD collision; preserve the established
local compilation path. Extend the existing halo/closed-wall AD probe to native
controllers with addressable inputs and explicit State arguments, retaining both
slip choices and its entire halo/JVP/VJP prelude.
Preserve every equation, reference archive and numerical tolerance; verify
primal/AD and native CPU/GPU behavior again before completing the final job and
commit. This is a correctness prerequisite discovered during authorized testing.
