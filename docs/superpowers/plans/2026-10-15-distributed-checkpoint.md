# Distributed checkpoint communication implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep checkpointed distributed derivatives from replaying halo communication while preserving scientific assertions and release gates.

**Architecture:** A shared checkpoint wrapper retains permutation results for multi-process execution. Both pure and scheduled rollouts use it. Local execution keeps JAX's existing checkpoint policy.

**Tech Stack:** Python >=3.12, JAX >=0.11.2, pytest, Sphinx.

**Spec:** `DESIGN.md`; the 2026-10-15 distributed-derivative investigation in `CHANGELOG.md`; GitHub failures 38714846084 and 38724977700.

## Global constraints

- Work on `jax-only`; commit only after the full correctness suite passes.
- No explicit test deadlines, skipped assertions, changed tolerances or reference data.
- Preserve checkpointing, ordinary asynchronous VJP, JVP and finite differences.
- Require successful GitHub checks before merging or publishing v0.2.0.

## Review focus

- Local CPU/GPU policy must stay unchanged: compare local checkpoint and direct derivatives.
- Multi-process policy must retain genuine adjoint exchanges: inspect compiled backward communication and run the native coupled probe.
- Checkpoint-disabled and zero-step execution must retain existing behavior: existing rollout tests.
- Scheduled output must share the policy without changing reduction semantics: existing output/gradient tests and GPU output checks.
- Both precision modes and wall-slip choices must remain correct: native GPU precision probes and the complete distributed halo prelude.

### Task 1: Retain distributed communication during checkpointing

**Files:**
- Create `veris/execution/checkpoint.py` and `tests/execution/test_checkpoint.py`.
- Modify `veris/execution/rollout.py`, `veris/io/scan.py`, `tests/closed_y_sharding_probe.py`.
- Update checkpoint documentation and `CHANGELOG.md`.

**Interfaces:**
- Consumes a pure `Callable[..., Any]` and JAX's static process count.
- Produces `checkpoint_transition(transition: Callable[..., Any]) -> Callable[..., Any]`, used by both rollout implementations.

- [x] Write the focused regression before implementation: actual derivatives agree with direct execution, while the distributed policy avoids replayed permutation instructions. The direct numerical function and JAX checkpoint baseline are independent comparison paths.
- [x] Run the regression and record its failure before adding the wrapper.
- [x] Implement the wrapper with a multi-process policy that saves `ppermute` results. Preserve the default local policy. Replace both direct checkpoint call sites.
- [x] Strengthen the existing native coupled probe with fresh VJP compilation coverage, retaining all original fields, JVPs and finite differences.
- [x] Run focused tests, inspect backward HLO for retained adjoint and absent replayed exchanges, and complete the scratch repeat controls. Any remaining mismatch blocks adoption.
- [x] Run maintained lint, formatting, annotations, ty, strict Sphinx and the full CPU suite once on final executable sources; run native GPU precision checks.
- [ ] Update the journal, obtain independent review, and commit/push the verified fix. Require fresh successful GitHub checks before the already-authorized release sequence.
