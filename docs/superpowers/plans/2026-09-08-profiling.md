# CPU and GPU profiling and optimization

Final status: implementation and CPU/GPU validation are complete. The sections
below preserve the exploratory plan and historical checkpoints; they are not
acceptance claims. See [the final report](../../../benchmarks/RESULTS.md).
The accepted API keeps Python `step` as the default and exposes opt-in
`compiled_step`; the EVP barrier applies only on CUDA. Full final suites passed
519 cases on each backend. CPU placement benefits are workload-dependent and
the cause of small-grid timing variability remains unresolved. All runs were
local, without scheduler jobs.

Objective: profile and optimize Veris on CPUs and GPUs with XProf/Perfetto
and TensorBoard in `.venv-latest`, retaining float64 physics and AD correctness.
Work starts from `jax-only` at `bce3b18`; do not use the old main branch.

## Initial measurements

The initial workload calls `veris.setup.artificial.initialize(64, 64)` and
replaces `nEVPsteps` with the model default of 400. All other artificial
settings and forcing retain their defaults. These are single-device runs;
the availability of two GPUs does not make this a distributed measurement.

JAX and jaxlib are 0.11.1. Installed TensorBoard 2.21.0 and XProf 2.23.1 into
`.venv-latest`; `pip check` passes. Hardware: Tesla P100-PCIE-16GB,
driver 580.173.02. GPU runs require access outside the filesystem sandbox.

Each process blocks on initialized state, records a first call separately,
then times 12 calls using `time.perf_counter` and `jax.block_until_ready`
on the entire returned PyTree. Calls reuse the same input state. Three
additional steps are profiled with `StepTraceAnnotation('coupled_step')`
and `jax.profiler.trace(..., create_perfetto_trace=True)`; trace overhead
is excluded from timing samples. No persistent compilation cache was configured.

| Backend | Existing step median | External whole-step JIT probe median |
| --- | ---: | ---: |
| CPU | 186.404 ms | 183.739 ms |
| One P100 | 27.900 ms | 20.701 ms |

The experimental callable is `jax.jit(step, static_argnames=['sett'])`.
Every returned state field matched the original step exactly for this input
on each backend (maximum absolute difference zero). This is exploratory
evidence only: separate processes, no paired randomized trials, no larger-grid
or gradient certification yet. The CPU difference does not establish a speedup.

Raw samples, XPlane protobufs and Perfetto JSON traces are in the untracked
`test_logs/profiling/{baseline,fused-probe}-{cpu,gpu}-64` directories.
First-call times include compilation and execution, not isolated compilation.
The GPU baseline trace records 294 executable dispatches over three steps.
The host calls `fill_overlap` on all 84 fields. CPU host event durations
include waits on earlier asynchronous work and cannot identify halo computation
as the dominant cost. Nested trace durations must not be summed as wall time.

TensorBoard serves these four profiles at `http://127.0.0.1:6006/` with the
XProf plugin enabled. The JSON discovery response is saved in
`test_logs/profiling/xprof-runs.json`. Local server session at this writing:
79737; verify its live handle before relying on it or restarting it.
TensorFlow is not installed; XProf reports some remote capture features disabled.
Local programmatic capture succeeds without TensorFlow.

## Work remaining

1. Build a separate reproducible benchmark harness, with tests before code.
   Explicit backend assertions, float64, grid/EVP controls, synchronization,
   isolated first-call/steady timings, JSON metadata, and optional traces.
   Include evolving coupled runs and fixed-state kernels; keep measurements
   out of pytest. Record CPU model, affinity, versions, revision and settings.
2. Measure 64x64 and larger grids (initially 256x256), with paired/interleaved
   candidates after warmup. Profile EVP and growth as well as the coupled step.
   Inspect device events through XProf and Perfetto, including kernel/launch
   counts and memory data. Verify XProf data conversion, not merely discovery.
3. Treat whole-step JIT as the first bounded candidate: preserve the public
   step signature and typed JIT convention. Add baseline comparison tests
   before implementation, covering multiple steps, masks, nonuniform inputs,
   scalar forcing changes and JVP/VJP equivalence. Retain only measured gains.
4. Investigate the actual CPU EVP kernels; whole-step JIT alone has not
   demonstrated a CPU improvement. Preserve iteration counts and equations.
5. Validate final changes on CPU and GPU, full correctness suite and maintained
   lint/type/coverage gates before committing to jax-only. Document benchmark
   commands, artifact locations, measured tradeoffs and limits. Review changes
   and inspect GitHub CI after authorized integration.

References: https://docs.jax.dev/en/latest/201/profiling.html and
https://openxla.org/xprof/capturing_profiles (consulted 2026-09-08).

## Implementation checkpoint

- Added `benchmarks/profile_veris.py` with tested paired fixed/evolving trials,
  explicit device selection and trace capture. Its baseline is the current
  unwrapped step body; use the saved old checkout to measure historical changes.
- Added typed whole-step JIT, keeping cooling dynamic. All-field evolving
  comparisons and cooling JVP/VJP/finite differences pass on CPU.
- Inserted an identity optimization barrier on drag/stress-divergence fields
  in EVP after capturing original-source oracles. Eight cases compare five
  result fields and smooth JVP/VJP against those snapshots and finite differences.
- CPU64 EVP probe: 190 to 118 ms. CPU256: no established speedup (1.50 to
  1.47 s within noise). GPU64 EVP probe: slight regression, 17.9 to 18.5 ms;
  GPU256: 66.1 to 54.5 ms. These are fixed-state solver measurements, not
  historical coupled-step speedups. Final combined profiles remain pending.
- XProf successfully converted original and whole-step-probe GPU kernel stats
  and CPU HLO stats through TensorBoard. Raw Perfetto data was parsed as well.
- Added optional profiler requirement pins and expanded maintained CI checks.
  Full maintained lint/format/annotation/ty pass. Independent review found no
  production blocker. Full CPU/GPU validation and final measurements pending.
