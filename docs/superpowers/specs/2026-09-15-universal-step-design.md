# Universal scan-based integration

## Status and scope

Approved by the user on 2026-09-15. Implement a shared public `veris.step`
for rollouts using `jax.lax.scan`, with optional `jax.checkpoint` on the scan
body. Support every setup under `veris/setups`, including sharded dynamics and
states initialized from external ocean geometry. Preserve existing numerical
equations, forcing conventions, diagnostics, and netCDF output semantics.

## Interface and ownership

Place integration in `veris/integration.py` and export `step` from `veris`.
Accept an initial State, a pure bound single-step callable, a static nonnegative
step count, and a checkpoint switch (enabled by default). Existing setup kernels
remain the authorities for their distinct physics sequences; callers bind their
configuration, physical constants, and optional cooling into the callable.
The default return is the final State, without storing a full State trajectory.

Support optional time-indexed forcing as a PyTree with a shared leading time
dimension; in this mode the callable receives State and one forcing slice.
Support an optional pure observation function that selects a small array PyTree
from each updated State. When supplied, return final State and stacked selected
observations. Initial-state observations remain explicit and outside the scan.
Validate count, forcing lengths, and callable contracts on the host where
possible. Shapes and PyTree structure must remain fixed through each rollout.
Zero steps return the initial State and correctly shaped empty observations.
JAX traces pure transitions/observers even for zero steps to infer output shapes.
An explicit `has_aux=True` mode accepts `(State, Diagnostics)` transitions and
`observe(state, diagnostics)`, keeping diagnostics outside the scan carry.

## Checkpointing and AD

Wrap the transition in `jax.checkpoint` when requested. Use the same transition
for checkpointed and ordinary scans. This recomputes transition intermediates
during reverse AD; it does not promise constant memory with rollout length.
Preserve gradients through initial State and dynamic forcing. Configuration and
physical constants retain their existing static treatment. Do not put host I/O,
NumPy conversions, or mutable output managers inside transformed code.

## Setup integration and output

Migrate dynamics and growth CLI integration and the parallel timing runner to
the shared scan driver. Keep setup-level single-step APIs for compatibility and
independent numerical comparisons. Island coupled integration uses the same
driver; ocean initialization is tested with a supplied physics kernel because
it defines geometry rather than a separate time integration scheme.

For CLI output, execute bounded scan chunks, collecting only fields required by
configured output. Replay selected per-step samples into the host output manager
with their exact elapsed times. This preserves complete-window averaging and
sampling while bounding history allocation. Sharded chunks keep the existing
mesh and halo exchange contracts. Timing warmup must not advance the actual
initial State and must compile every scan length being measured, including a
short final chunk. Histories retain storage halos until host collection; their
new leading time axis is unpartitioned.

## Alternatives

1. Recommended: generic scan driver over existing physics callables. Supports
   all setups without changing their distinct forward equations.
2. A mode-switching monolithic physics step would duplicate setup policy and
   couple the integration layer to every experiment.
3. Independent scan loops per setup would retain duplicated rollout and AD
   behavior and would not provide the requested universal API.

## Verification gates

- Test before implementation: compare every State field against explicit
  repeated setup steps for coupled island, dynamics, growth, and
  ocean-initialized runs; include zero, one, and multiple steps.
- Compare checkpoint on/off values, JVPs, and VJPs; compare nontrivial weighted
  objectives with finite differences for initial-state and forcing sensitivity.
  Exercise time-varying cooling, both precisions, and relevant masked boundaries.
- Inspect JAX programs to establish scan and checkpoint use, independently of
  numerical equivalence.
- Exercise actual sharded execution and AD, with serial comparisons; run CPU
  and available GPU setup paths, including parallel CLI netCDF output.
- Verify sampling times, averaging, final snapshots, zero-step output, and
  unchanged warmup semantics after CLI migration.
- Run one pytest process at a time, focused/fast tests during development and
  the full correctness/coverage suite before committing. Run maintained lint,
  formatting, type checks, and a Sphinx build. Update CHANGELOG with evidence.
- Obtain independent code/test review and commit passing work to `jax-only`.
