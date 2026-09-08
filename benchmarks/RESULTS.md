# Local CPU/GPU profiling results — 2026-09-08

The accepted implementation retains the Python `artificial.step` driver, adds
the opt-in `artificial.compiled_step`, and materializes shared EVP drag and
stress-divergence intermediates on CUDA only. CPU fusion is unchanged. Equations,
precision and iteration counts are unchanged. Whole-step compilation is useful
on the measured GPU; it is not a general CPU optimization.

## Measurement conditions

All runs used the local node without scheduler jobs. Hardware was a Xeon
E5-2650 v4 host with 24 physical/48 logical CPUs and four NUMA nodes, and one
Tesla P100-PCIE-16GB. The environment was `.venv-latest`: JAX/jaxlib 0.11.1,
NumPy 2.5.3, TensorBoard 2.21.0, XProf 2.23.1 and Perfetto Python 0.58.2.

Final measurements used the artificial masked sea, float64, 400 EVP iterations,
three warmups and twelve evolving steps. Each call synchronized its entire
output. Python and compiled variants alternated AB/BA; first-call and final
outputs were checked outside timing (`--validation final`). All checked fields
matched exactly within each final paired run. These checks supplement the
historical-source primal/gradient tests; they do not independently establish
cross-revision equivalence. First-call compilation and separate trace capture
are excluded from the medians below. See [README.md](README.md) for commands.

The historical source was archived from `76bf8bc`, with the same benchmark
harness copied into that checkout. GPU original/current rows are separate
processes, not randomized historical pairs. Similar earlier measurements support
the direction of the GPU result; these samples do not establish statistical
confidence intervals or performance on other hardware.

## GPU coupled-step latency

| Interior grid | Historical Python | Historical compiled | Current Python | Current compiled |
| --- | ---: | ---: | ---: | ---: |
| 64 × 64 | 26.35 ms | 20.40 ms | 26.65 ms | 20.91 ms |
| 256 × 256 | 74.67 ms | 69.13 ms | 62.74 ms | 57.48 ms |

The combined current compiled path is about 21% lower latency at 64 × 64 and
23% lower at 256 × 256 than historical Python. The CUDA barrier itself slightly
costs performance at the smaller grid; the compiled driver still gives a net
gain. At the larger grid both changes contribute. Keep the opt-in compiled
callable outside the integration loop and reuse it.

## CPU placement and variability

| Grid / affinity | Python medians, two runs | Compiled medians, two runs |
| --- | ---: | ---: |
| 64 × 64 / all 48 logical CPUs | 126.87, 200.16 ms | 160.33, 183.13 ms |
| 64 × 64 / socket 0, 12 physical cores | 168.42, 171.36 ms | 156.18, 159.77 ms |
| 256 × 256 / all 48 logical CPUs | 1618.25, 1583.20 ms | 1636.45, 1616.35 ms |
| 256 × 256 / socket 0, 12 physical cores | 1492.68, 1460.65 ms | 1511.36, 1486.96 ms |

Restricting this process to one socket lowered the larger-grid Python medians
by about 8% in both rounds. This is a local placement result, not a code speedup
or a universal affinity recommendation. No stable small-grid CPU gain is claimed.
An earlier uninterrupted Python-only loop favored socket placement, but paired
measurements do not consistently reproduce that ordering. Those methods also
differ in device context, resident compiled variants and call interleaving;
the cause remains unresolved. Benchmark the actual integration loop before
choosing CPU affinity or the compiled driver.

During the final serial matrix, 540 one-second `vmstat` intervals had median
95% host-wide idle CPU, minimum 76%, maximum 1% I/O wait and zero steal.
Other users can affect timings, including on a mostly idle host through shared
cores, memory, cache and frequency effects. These observations neither establish
contention as the cause nor rule out localized contention or earlier heavy use.
No other users' processes were stopped or modified.

## Profiling and correctness evidence

TensorBoard's XProf plugin discovered and converted GPU kernel statistics.
Perfetto's pinned v57.2 trace processor parsed the timelines and executed SQL.
The earlier original/optimized 256-grid GPU traces show 294 versus 3 executable
dispatches across three coupled steps; GPU kernel counts were 36,279 versus
34,728. These are separate traced runs, not the unprofiled latencies above.
CPU XProf HLO statistics tables were empty; CPU analysis instead used runtime
timeline events and compiled HLO. Nested host durations were not summed as
wall time. Peak memory and distributed scaling were not measured.

Both final full correctness suites passed: 519 tests on CPU and 519 with GPU
as the default backend. Tests include masked evolving coupled states,
fixed/adaptive 400-step historical EVP oracles, JVP/VJP and independent finite
differences. Maintained coverage was 1041/1050 statements (99.14%); whole-package
coverage was 74.20%. Maintained Ruff, format, annotation and type checks passed,
as did `pip check`. No tolerance was relaxed.

Rejected approaches include a blanket whole-step JIT default and CPU-wide EVP
barriers: exploratory gains did not survive representative native CPU driver
measurements. Basal-factor hoisting and concatenated halo experiments did not
justify changes. Do not use their earlier timings as accepted CPU improvements.

## Local artifacts

Artifacts remain under `test_logs/profiling/`, outside version control:

- `accepted-*/results.json`: raw final samples, settings, versions and affinity;
  accompanying `interval.json`, logs and first-round XPlane/Perfetto traces.
- `accepted_matrix.py` and `accepted-vmstat.log`: serial local runner and load log.
- `final-full-{cpu,gpu}.log`, `final-coverage.json`: final validation.
- `perfetto-sql-summary.json`, `xprof-final-kernel-summary.json`: trace analyses.
- `native-affinity-*`: earlier standalone placement experiment and exact outputs.
- `final-source-sha256.json`, `final-source.patch`: measured source provenance.

The small historical numerical oracle fixtures are versioned with their source
hashes under `tests/reference_data/`. Performance traces are too large to commit.
