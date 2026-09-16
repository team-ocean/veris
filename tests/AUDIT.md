# Test suite redundancy audit — 2026-09-16

## Decision rule

A second call is redundant when its inputs, branch and assertions establish
an already-owned contract. Shared source lines alone do not establish that:
independent equations, precision boundaries, AD directions, masks, distributed
communication and actual CLI composition remain separate requirements.

Baseline: `4246f23`, 1,059 collected cases (1,057 CPU passes, two GPU-only skips).
All baseline source/test hashes match the recorded full-run artifacts. Baseline
maintained coverage is 2,668 / 2,746 statements (97.16%); whole-package coverage
is 2,668 / 3,099 (86.09%). No production source, tolerance relaxation, test skip,
collection exclusion or coverage exclusion is introduced by this audit.

Compacted collection: **950 cases**, a reduction of **109 (10.3%)**. This also
removes repeated work inside surviving cases; collection count understates the
execution reduction. Full verification is recorded in CHANGELOG.md.

Full CPU verification passed: **948 passed, two expected GPU-only skips**.
Every maintained source file has exactly the same executed-line set as the
baseline, and all source/test hashes match the tested snapshot. Maintained
coverage remains **97.16%**, whole-package coverage **86.09%**.

| Measure | Baseline | Compacted |
| --- | ---: | ---: |
| Collected cases | 1,059 | 950 |
| Test modules | 66 | 65 |
| Setup forward scan calls | 96 | 16 |
| Maintained statements covered | 2,668 / 2,746 | 2,668 / 2,746 |
| Recorded full CPU wall time | 1,951.70 s | 1,654.90 s |

The recorded full run is 296.80 seconds shorter (15.2%). This is end-to-end
test timing; solver performance remains in the separate benchmark harness.
Focused CUDA verification passed **211 tests with no skips**, including both
GPU-only cases omitted by the CPU run. Maintained Ruff, formatting, annotation
and ty checks also pass. Local evidence is in `test_logs/test-compaction/`,
including baseline coverage, source hashes, CPU/GPU XML and exact-line comparison.

## Removed work and retained owners

| Area | Removed or reformulated work | Retained contract |
| --- | --- | --- |
| Basal drag | 30 duplicate primal cases | `test_basal_drag_value_and_gradients_match_stable_keel_law` gets full-array primal output from `value_and_grad(has_aux=True)`. All 30 dtype/area/thickness combinations and both analytic sensitivities remain. |
| Area and mass | Nine random-seed repetitions | `test_area_and_mass_staggering` retains both rectangular orientations, singleton axis, independent periodic indexing and mass conservation. The source map is linear and has no value-dependent branch. |
| Corner averaging | Ten random-mask repetitions | `test_corner_average_with_land` packs all 16 binary four-cell coast orientations into one rectangular field under each slip mode. Independent full-grid indexing, all-land and all-ocean checks remain. |
| Mass AD | Four thickness repetitions | `test_mass_gradient` retains both density derivatives and finite differences; this linear map has constant sensitivities. |
| Cleanup | 15 scalar cases become one array call | `test_cleanup_thresholds_and_overshoots` retains the full 5-by-3 ice/snow Cartesian product and every independent scalar expected value. The kernel is pointwise. |
| Limiter | Eight scalar primal cases | `test_superbee_piecewise_slopes_and_selected_kink_linearizations` owns values in every region, all breakpoints, JVP/VJP and selected kink slopes. Separate one-sided checks remain. |
| Transport | Duplicate zero-velocity axis row | `test_cfl_one_is_exact_periodic_translation` retains zero flow once, and both axes/signs for nonzero flow. At zero velocity the two removed/retained inputs were identical. |
| Atmosphere helpers | Three wind cases become one array call | `test_drag_and_neutral_stability_functions` retains all three drag values and checks identical neutral stability limits once. |
| Hydrostatic tilt | Two inactive freshwater-switch repetitions | `test_affine_hydrostatic_tilt_and_wind_force` retains both switch settings for ice load, where it changes behavior, and independent elevation/pressure cases. |
| EVP reference AD | Duplicate invocation of the same pullback | `test_evp_optimization.py` reuses its already calculated reverse derivative; assertions are unchanged. |
| Schema and State | Two cases and the redundant `test_state.py` module | `test_registry_defaults_are_disjoint_complete_and_frozen` owns registry/default/schema and compatibility alias checks. `test_initialize_allocates_complete_minimal_frozen_state` now owns explicit PyTree roundtrip and leaf identity. Existing initialization JIT/JVP/VJP checks retain immutable replacement behavior. |
| Central type ownership | Two duplicate cases; repeated subprocess assertions | `test_registry_types_have_local_owners` and `test_registries_share_parameter_metadata` own these contracts. Fresh-process import-order and default smoke checks remain. |
| Coefficient registries | Repeated instance/default correspondence and description assertions | Configuration's complete registry test owns these; `test_scattered_coefficient_defaults_match_original_literals` retains independent literal expectations. |
| Ocean initialization | Two separately allocated cases | Nonuniform corner-area indexing moves into `test_initialization_surface_masks_and_reciprocals`. Configured surface temperature moves into `test_external_fields_and_static_overrides`, retaining both precisions. The nonuniform area oracle subsumes constant averaging. |
| Static typing | Four positive checker subprocesses | Dynamics and directional transport each put both valid signatures in one module; solver valid iteration count joins positive solver contracts; height helper joins the valid bulk helper module. Separate negative type/arity contracts remain. |
| Artificial forcing | One separate test and two complete trajectories | `test_default_cooling_and_prescribed_forcing_replace_previous_outputs` checks implicit default against explicit 100 with overwritten prior fluxes, asserting exact equality of every field. Nondefault cooling sensitivity remains covered elsewhere. |
| Snapshot I/O | One duplicate nonuniform-halo snapshot | `test_full_snapshot_roundtrip_and_selected_physical_snapshot` already checks physical cropping and selected/nonuniform data. |
| Scheduled output | One separate compiled trajectory | `test_collectors_receive_only_completed_reductions_and_instant_records` now also rejects direct `sample` replay while verifying reduced values and collection count. |
| Output schedule | One duplicate invalid-sampling case | `test_incompatible_sampling_rejected_before_iterator_consumption` checks eager rejection at both zero and positive lengths. |
| Output validation | 16 checkpoint rejection rows become four | `test_invalid_checkpoint_fails_before_compilation_or_output` retains every invalid value and every zero/positive, scheduled/generic combination. The guard runs before dispatch. |
| Growth CLI | One duplicate zero-step invocation | `test_growth_cli_zero_steps_saves_initial_column` now writes to missing nested directories, preserving time, shape, values and parent creation. |
| Setup forward rollouts | 96 scan calls become 16 across eight setup/dtype cases | `test_setup_rollout_matches_explicit_steps` retains all-field, three-step and diagnostic comparisons under both checkpoint modes. `test_integration.py` owns zero/one-step, observed/unobserved, forcing and auxiliary-return contracts against independent recurrences. The comparison helper now checks exact leaf shapes. |
| Setup AD | Two ocean derivative cases | `test_real_rollout_state_and_forcing_derivatives` retains artificial/dynamics/growth in both precisions with state/forcing JVP/VJP/FD and both checkpoint modes. Ocean and artificial use the same coupled advance; geometry initialization occurs outside differentiation. Float64 inputs match exactly; float32 dxV/dyV differ only by one rounding step (relative 6.1e-8). Both ocean forward precision cases and dedicated initialization tests remain. |

Numerical changes remove 80 cases; schema/type/initialization changes remove
11; output changes remove 15; scenario/rollout changes remove three. No deleted
test is retained under an opt-in marker or moved to an unexecuted archive.

## Reviewed and deliberately retained

All 66 baseline `test_*.py` modules were inspected, including both setup modules.
The table covers changed files. The following groups retain distinct contracts:

- `test_ad_freedrift`, `test_ad_primitives`, `test_ad_sharding`,
  `test_ad_thermodynamics`, `test_ad_zero_states`, `test_nonsmooth_gradients`:
  selected origin linearizations, inactive inputs, kink conventions, true
  nonlinear sensitivities and partitioned AD cannot be inferred from primal tests.
- `test_dynamics_routines`, `test_evp_solver`, `test_freedrift_solver`,
  `test_growth`, `test_heat_flux_MITgcm`, `test_ocean_stress`, `test_solve4temp`:
  independent equations, masks and solver branches remain meaningful even where
  integration tests also execute those kernels.
- `test_precision`, `test_precision_paths`, `test_surface_temperature_precision`:
  float32 overflow/underflow, scalar policy, diagnostic and compiled-path dtypes
  are distinct from float64 numerical agreement.
- `test_fill_overlap`, `test_initialized_sharding`, `test_global_sum`,
  `test_closed_y`, `test_closed_y_stress`, `test_parallel_case` and their probes:
  partition packing, global reductions, real subprocess communication, closed
  walls, wall stress and public launch behavior have separate oracles. Serial
  agreement alone cannot establish correct communication or boundary values.
- `test_dynamics_case`, remaining `test_growth_case`, `test_diagnostics`,
  `setups/test_compiled_step`, remaining `setups/test_artificial`:
  reference composition, recursive heat fluxes, standalone import/CLI behavior,
  exact diagnostics separation and compiled/eager agreement remain distinct.
- `test_integration`, remaining `test_rollout_setups` and
  `rollout_parallel_probe`: independent recurrence arithmetic, rematerialization,
  evolving forcing, real multi-step physics and distributed scan derivatives
  each catch different failures. Both device precisions remain.
- `test_variables`, `test_typing_contracts` and remaining schema tests: metadata,
  public type contracts, validation and registry-to-instance fidelity remain.
- `test_io_calendar`, `test_io_distributed`, `test_io_integration`,
  `test_io_output`, `test_io_restart`, `test_output_reduced`,
  `test_output_scan_memory`, `test_output_scan_parallel`,
  `test_output_scan_physics`, `test_output_scan_validation` and remaining storage,
  schedule, scan and rollout-output tests: calendars, restart equivalence,
  schema validation, direct versus reduced writer lifecycle, bounded storage,
  real physics precision and distributed collection are separate contracts.
- `test_benchmark_profile`: paired trajectory isolation, validation scheduling,
  synchronization and real trace output test the benchmark tool itself.

## Maintenance policy

Before adding a test, identify the contract and its current owner. Put another
assertion on an existing run when setup, inputs and trajectory are identical.
Batch independent pointwise examples when this retains a readable scalar oracle.
Parameterize distinct branches and boundaries, not arbitrary repetitions of a
linear map. Keep negative validation cases separate when a prior failure could
hide a missing check. Preserve explicit independent reference calculations.
