# Coupled-step profiling

See [the local measurement report](RESULTS.md) for accepted changes, results,
correctness evidence and unresolved CPU timing variability.

Use the project root and `.venv-latest`. The workload is the artificial periodic
sea with a central island, prescribed forcing, float64 arrays and 600-second
steps. Grid arguments count interior cells; each dimension adds four halo cells.
The benchmark changes the artificial example's five EVP iterations to an explicit
count, defaulting to 400. It does not establish convergence of that count.

The model exposes two drivers with the same physics: `artificial.step` retains
the Python integration sequence, while `artificial.compiled_step` compiles the
whole sequence. Choose the callable once before an integration loop. Whole-step
compilation reduces GPU dispatch overhead but is not consistently faster on CPU.
CPU placement can matter even on an otherwise idle multi-socket host; the local
affinity example below changes only the launched process.

```bash
source .venv-latest/bin/activate
python -m pip install -r requirements-profile.txt
JAX_PLATFORMS=cpu python -m benchmarks.profile_veris --backend cpu --nx 64 --ny 64 --evp-steps 400 --repeats 12 --warmup 3 --mode fixed --output test_logs/profiling/paired-cpu-64-fixed --trace
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.profile_veris --backend gpu --nx 64 --ny 64 --evp-steps 400 --repeats 12 --warmup 3 --mode fixed --output test_logs/profiling/paired-gpu-64-fixed --trace
```

Repeat with `--nx 256 --ny 256`, and with `--mode evolving`, using a new output
directory each time. Run performance measurements serially on otherwise idle
hardware, separately from pytest. CPU affinity is recorded but not set by the
harness; use the same permitted affinity and thread environment when comparing
runs. GPU measurements use the first visible GPU and require actual GPU access.
Unset `JAX_PLATFORMS=cpu` before a GPU run if it was exported in the shell.
The requested backend is selected explicitly and checked; unavailable GPU access
raises an error instead of silently reporting CPU timings.


For CPU placement experiments, restrict only the benchmark child process, before
JAX initializes. Inspect `lscpu -e=CPU,NODE,SOCKET,CORE` and
`taskset -pc $$` first; choose CPUs within the process's allowed set. The following
socket-zero physical-core list applies to the measured local host, not arbitrary
machines. Compare against an unpinned run with otherwise identical arguments.

```bash
JAX_PLATFORMS=cpu taskset -c 0,2,4,6,8,10,12,14,16,18,20,22 python -m benchmarks.profile_veris --backend cpu --nx 64 --ny 64 --evp-steps 400 --repeats 16 --warmup 4 --mode evolving --validation final --output test_logs/profiling/paired-cpu-64-socket0-final
```

Use `each` to validate every intermediate state and `final` to corroborate timing
without validation gaps. Preserve both schedules' results and repeat fresh
processes; do not compare an `each` baseline against a `final` candidate.

Each directory contains `results.json`. With `--trace`, `baseline/` and
`candidate/` each contain an XPlane protobuf and compressed Perfetto JSON under
`plugins/profile/`. These are three additional calls after timing, so trace
capture overhead is excluded from the recorded samples. Evolving traces restart
from the initial state; they are not a capture of the final measured steps.

```bash
source .venv-latest/bin/activate
tensorboard --logdir test_logs/profiling --host 127.0.0.1 --port 6006
```

Use TensorBoard's Profile tab with the installed XProf plugin. Open the generated
`perfetto_trace.json.gz` in Perfetto for timeline inspection. Trace file creation
alone does not prove that all XProf analyses are available: inspect the desired
analysis and its device events before drawing conclusions.

For scripted analysis, the validated environment also has Perfetto 0.58.2 and
its pinned v57.2 trace processor in `.venv-latest/bin/trace_processor`. Use the
[Perfetto Python API](https://perfetto.dev/docs/analysis/trace-processor-python)
with `TraceProcessorConfig(bin_path='.venv-latest/bin/trace_processor')` to keep
the tool inside this environment. For example, this SQL counts the executable
dispatch events in a GPU trace:

```sql
SELECT name, COUNT(*) AS occurrences, SUM(dur) / 1e6 AS total_ms
FROM slice
WHERE name = 'PjRtCApiLoadedExecutable::Execute'
GROUP BY name;
```

## What is compared

`baseline` calls the Python body of the current `artificial.step`; if the public
function is decorated, its `__wrapped__` body is used. `candidate` uses the public
`artificial.compiled_step` when available; on historical checkouts it applies a
fresh `jax.jit(..., static_argnames=['sett'])` to the Python body. Both variants
use the same current physics kernels and settings. This isolates the whole-step
compilation boundary. It is **not a comparison against the historical source**,
and cannot establish the gain from changes made inside EVP, growth or other
kernels. Preserve results from the previous revision and measure those changes
separately with matching hardware, workload and runtime settings.

Initialization is synchronized before measurement. Each timed call blocks on the
entire returned PyTree. First calls are reported separately, followed by fixed
input warmup calls. Measured pairs alternate baseline/candidate and
candidate/baseline. Fixed mode always reuses the initial input; evolving mode
starts both independent trajectories from the initial state after warmup.

The default `--validation each` compares every measured pair on the host outside
the timer. `--validation final` compares first-call and final measured outputs
only, avoiding host copies and array reads between timed pairs. It cannot detect
intermediate discrepancies that later disappear. Both schedules compare all
fields and reject NaN/Inf in checked outputs, using `rtol=1e-10, atol=1e-12`.
The JSON records the schedule, number of comparisons, and largest absolute error
across those comparisons. Neither schedule establishes physical correctness or
gradient equivalence. Final mode still synchronizes every call and alternates
variants; it does not measure asynchronous integration throughput.

## Interpretation limits

- In `each` mode, host validation copies GPU outputs and reads CPU arrays between
  pairs. These
  untimed operations can affect caches, memory pressure, device clocks and gaps
  between launches. Alternating order reduces simple position bias but does not
  eliminate it. These are isolated synchronized step latencies with validation
  gaps, not uninterrupted integration throughput. Corroborate small gains with
  repeated independent runs and an uninterrupted workload before accepting them.
- First-call times include compilation and execution. Baseline runs first, and
  the candidate can share cached inner compilations. Existing in-process or
  persistent cache entries can also be reused. These numbers are not isolated
  cold compilation measurements; use separate fresh processes and controlled
  cache configuration for that question.
- `median_paired_speedup` is the median of corresponding baseline/candidate
  sample ratios; it can differ from the ratio of the two reported medians.
  Raw samples and call order are retained. No confidence interval or significance
  test is supplied; a ratio slightly above one is not proof of an improvement.
- Metadata records revision, whether tracked files were dirty, versions, CPU
  model/affinity, device, environment controls and all model settings. A revision
  with `tracked_dirty=true` does not identify the patch. Untracked source files
  are not included in that flag. Preserve the exact patch or commit alongside
  measurements for reproducibility; use a new output directory per run because
  `results.json` is overwritten if the directory is reused.
- The CLI is intended for a fresh process. Reusing `main()` in a process with
  previously imported distributed halo modules or preconfigured JAX can retain
  import/cache state. `initialize()` selects serial halos; these are single-device
  measurements and provide no multi-device scaling evidence.
- The harness reports coupled-step latency. It does not separately time EVP or
  growth, measure peak memory, isolate individual kernel costs, or compare
  numerical algorithms. Attribute costs using the captured device timeline and
  separate kernel experiments; nested host durations must not be summed as wall
  time.
