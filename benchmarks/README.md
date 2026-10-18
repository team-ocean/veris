# Coupled-step profiling

See [the local measurement report](RESULTS.md) for accepted changes, results,
correctness checks and the unresolved variation in CPU timings.

Use the project root and `.venv-latest`. The benchmark runs the island setup:
a sea with a central island and periodic boundaries that connect opposite edges.
It uses supplied forcing (inputs such as wind and heat), float64 arrays (64-bit
numbers) and 600-second steps. Grid arguments count interior cells. Each dimension
adds four halo cells, extra boundary cells that hold neighboring values. The
benchmark changes the island example's five EVP iterations to an explicit count,
defaulting to 400. EVP is the elastic-viscous-plastic method for calculating ice
motion. Each iteration repeats a calculation to improve the solution. The
benchmark does not show that this count is enough for the solution to converge.

The benchmark compares eager `veris.advance`, which runs without explicit
whole-step compilation, with an explicit `jax.jit` of the same coupled transition
and supplied heat forcing. A coupled transition advances both ice motion and
freezing/melting. JIT compiles the calculation for a computing device.
Choose the callable function once before the loop that advances the model.
Compiling the whole step reduces the overhead of launching work on a GPU, but
is not consistently faster on a CPU. CPU placement can matter even on an idle
host with several processor sockets. Affinity limits which CPUs a process can
use; the local affinity example below changes only the launched process.

```bash
source .venv-latest/bin/activate
python -m pip install -r requirements-profile.txt
JAX_PLATFORMS=cpu python -m benchmarks.profile_veris --backend cpu --nx 64 --ny 64 --evp-steps 400 --repeats 12 --warmup 3 --mode fixed --output test_logs/profiling/paired-cpu-64-fixed --trace
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.profile_veris --backend gpu --nx 64 --ny 64 --evp-steps 400 --repeats 12 --warmup 3 --mode fixed --output test_logs/profiling/paired-gpu-64-fixed --trace
```

Repeat with `--nx 256 --ny 256`, and with `--mode evolving`, using a new output
directory each time. Run performance measurements one at a time on otherwise
idle hardware, separately from pytest. On Linux, `cpu_affinity` records the sorted
allowed CPU IDs from `os.sched_getaffinity(0)`. The harness does not set affinity.
On macOS/Darwin and other platforms without that API, `cpu_affinity` is `null`.
The separate `cpu_count` field records the logical CPU count from `os.cpu_count()`
on all platforms, or `null` if unknown. It does not describe which CPUs the
process is allowed to use. Use the same permitted affinity and thread environment when comparing
runs. GPU measurements use the first visible GPU and require actual GPU access.
Unset `JAX_PLATFORMS=cpu` before a GPU run if it was exported in the shell.
The harness explicitly selects and checks the requested backend, the CPU or GPU
system used for calculations. If GPU access is unavailable, it raises an error
instead of silently reporting CPU timings.


For Linux CPU placement experiments, restrict only the benchmark child process
before JAX starts. Inspect `lscpu -e=CPU,NODE,SOCKET,CORE` and
`taskset -pc $$` first; choose CPUs within the process's allowed set. The following
socket-zero physical-core list applies to the measured local host, not arbitrary
machines. Compare against a run with no added CPU restriction and otherwise identical arguments.

```bash
JAX_PLATFORMS=cpu taskset -c 0,2,4,6,8,10,12,14,16,18,20,22 python -m benchmarks.profile_veris --backend cpu --nx 64 --ny 64 --evp-steps 400 --repeats 16 --warmup 4 --mode evolving --validation final --output test_logs/profiling/paired-cpu-64-socket0-final
```

Use `each` to check every intermediate state. Use `final` to check the timings
without pauses for validation between measured steps. Keep both schedules'
results and repeat in fresh processes. Do not compare an `each` baseline against
a `final` candidate.

Each directory contains `results.json`. With `--trace`, `baseline/` and
`candidate/` each contain an XPlane protobuf and compressed Perfetto JSON under
`plugins/profile/`. These trace files record a timeline of computing work. The
trace uses three additional calls after timing, so its recording overhead is
excluded from the measured samples. Evolving traces restart from the initial
state; they do not record the final measured steps.

```bash
source .venv-latest/bin/activate
tensorboard --logdir test_logs/profiling --host 127.0.0.1 --port 6006
```

Use TensorBoard's Profile tab with the installed XProf plugin. Open the generated
`perfetto_trace.json.gz` in Perfetto to inspect the timeline. Creating a trace
file does not prove that all XProf analyses are available. Inspect the analysis
you need and its device events before drawing conclusions.

For analysis with scripts, the checked environment also has Perfetto 0.58.2 and
its trace processor fixed at v57.2 in `.venv-latest/bin/trace_processor`. Use the
[Perfetto Python API](https://perfetto.dev/docs/analysis/trace-processor-python)
with `TraceProcessorConfig(bin_path='.venv-latest/bin/trace_processor')` to keep
the tool inside this environment. For example, this SQL counts events that launch
compiled calculations in a GPU trace:

```sql
SELECT name, COUNT(*) AS occurrences, SUM(dur) / 1e6 AS total_ms
FROM slice
WHERE name = 'PjRtCApiLoadedExecutable::Execute'
GROUP BY name;
```

## What is compared

`baseline` calls the shared `advance` with the island's supplied cooling and
zero shortwave radiation (incoming sunlight). `candidate` applies `jax.jit` to
that same State transition, with settings and constants fixed before measurement.
Both variants use the same current physics kernels, the functions that perform
the calculations, and the same settings. The comparison measures the effect of
compiling the whole step. It is **not a comparison against the historical source**,
and cannot show the gain from changes inside EVP, growth or other kernels. Keep
results from the previous revision and measure those changes separately with
matching hardware, workload and runtime settings.

Initialization finishes before measurement starts. Each timed call waits for
the entire returned PyTree, a nested structure of arrays, to finish computing.
First calls are reported separately, followed by warmup calls with fixed inputs.
Measured pairs alternate baseline/candidate and candidate/baseline. Fixed mode
always reuses the initial input. Evolving mode starts both independent
trajectories, sequences of states, from the initial state after warmup.

The default `--validation each` compares every measured pair on the host, the
Python process controlling the run, outside the timer. `--validation final`
compares only first-call and final measured outputs. This avoids host copies and
array reads between timed pairs. It cannot detect intermediate differences that
later disappear. Both schedules compare all fields and reject NaN/Inf (undefined
or infinite numbers) in checked outputs, using `rtol=1e-10, atol=1e-12`. The JSON
records the schedule, number of comparisons and largest absolute error across
them. Neither schedule proves physical correctness or equal gradients, which
describe how results change with inputs. Final mode still waits for every call
and alternates variants. It does not measure the rate of advancing the model
when calls are launched without waiting between them.

## Interpretation limits

- In `each` mode, host validation copies GPU outputs and reads CPU arrays between
  pairs. These untimed operations can affect caches, memory pressure, device
  clocks and gaps between launches. Alternating the order reduces bias from
  which variant runs first, but does not eliminate it. The measurements are times
  for individual steps that wait for completion, with validation pauses. They
  do not measure the rate of an uninterrupted run. Check small gains with repeated
  independent runs and an uninterrupted workload before accepting them.
- First-call times include compilation and execution. Baseline runs first, and
  the candidate can reuse cached compilations from inside the step. Existing
  cache entries in the process or on disk can also be reused. These numbers do
  not measure compilation from scratch in isolation. To measure that, use
  separate fresh processes and controlled cache settings.
- `median_paired_speedup` is the median of corresponding baseline/candidate
  sample ratios. It can differ from the ratio of the two reported medians. A
  median is the middle value in a sorted set, or the average of the two middle
  values when the count is even. Raw samples and call order are
  kept. The harness provides no confidence interval (an estimate of uncertainty)
  or significance test. A ratio slightly above one is not proof of an improvement.
- Metadata records the revision, whether tracked files had uncommitted changes,
  versions, CPU model/affinity, device, environment controls and all model
  settings. A revision with `tracked_dirty=true` does not identify the patch.
  The flag does not include untracked source files. Keep the exact patch or
  commit alongside measurements so others can reproduce them. Use a new output
  directory per run because reusing it overwrites `results.json`.
- The command line interface (CLI) is intended for a fresh process. Reusing
  `main()` in a process that already imported distributed halo modules or
  configured JAX can retain import/cache state. `initialize()` selects serial
  halos. These measurements use one device and give no evidence of how
  performance changes with more devices.
- The harness reports the time for a coupled step. It does not separately time
  EVP or growth, measure peak memory, isolate individual kernel costs or compare
  numerical algorithms. Use the recorded device timeline and separate kernel
  experiments to identify costs. Do not add nested host durations to calculate
  total elapsed time: they overlap.

## Spherical transport acceptance

Run `.venv/bin/python -m benchmarks.validate_spherical --steps 1000 --output test_logs/spherical.json`.
The harness advances a smooth positive tracer with the native extensive
advection operator under known zonal rotation. It checks stored-area volume,
finite values and bounds at every iteration, independently checks northward
face integrals, and measures tensor refinement at4/2/1degree spacing. A
Cartesian-face negative control conserves volume but fails the spherical
tendency test. Separate real thermodynamic probes verify retained intermediate
physics status. Transport does not invoke cleanup, ridging or thermodynamics.

Local float64 CPU evidence on the N3 science plus the recorded harness SHA:
1,000steps, maximum relative volume drift2.11e-15, tensor orders1.9997363 and
1.9999341. The JSON includes exact source commit, dirty flag, harness checksum,
geometry, dtype and actual runtime/device metadata. This is modeled numerical
acceptance, not an observational climate-skill assessment or a GPU result.
