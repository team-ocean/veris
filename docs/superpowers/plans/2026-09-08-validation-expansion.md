# Validation expansion implementation plan

Goal: verify MITgcm mask semantics, nonsmooth AD, multi-process reductions,
real GPU execution, and broader maintained repository style/type quality.

Design: extend the independent equation-based harness in DESIGN.md. Preserve
wet-cell equations; define land diagnostics as zero using safe inactive inputs.
Use explicit mesh collectives for local residuals, keeping serial calls valid.
Threshold tests distinguish branch derivatives from classical differentiability.
Actual GPU and process results are required, not availability skips.

- [x] Add mask regression tests with wet/land mosaics, invalid land input,
  and zero land sensitivities. Verify failure then fix bulk inputs/outputs.
- [x] Add nonsmooth forward/reverse derivative tests and one-sided limits.
- [x] Trace EVP local sums and mesh calling convention; add analytic serial
  and multi-process reduction tests before implementing missing collectives.
- [x] Install JAX 0.11.1 CUDA 12 for two P100s; require explicit GPU backend,
  run full correctness suite and two-device/multi-process communication probes.
- [x] Audit Ruff/ty; fix maintained code findings without concealing numerical
  changes in formatting. Keep generated metadata distinct in reports.
- [x] Run full CPU suite, coverage gate, lint/type checks, and review diffs;
  update CHANGELOG.md with exact results and remaining limits before committing.

Ownership: root mask/reductions/GPU; nonsmooth agent new gradient tests;
style_audit agent read-only quality audit. Only root runs pytest.
