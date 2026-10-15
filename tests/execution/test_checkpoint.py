"""Checkpointed communication retains correct derivatives without replay."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = """
import sys
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from veris.execution.checkpoint import checkpoint_transition

jax.config.update("jax_enable_x64", True)
mesh = Mesh(np.asarray(jax.devices()).reshape(2), ("y",))

def local_transition(value):
    def iteration(_, current):
        exchanged = jax.lax.ppermute(current, "y", [(0, 1), (1, 0)])
        return jnp.sin(current + 0.2 * exchanged)
    result = jax.lax.fori_loop(0, 2, iteration, value)
    outer = jax.lax.ppermute(value, "y", [(0, 1), (1, 0)])
    return result + 0.1 * outer ** 2

mapped = jax.shard_map(local_transition, mesh=mesh, in_specs=P("y"), out_specs=P("y"))
count = jax.process_count
try:
    # Select the host policy while keeping actual numerical device execution.
    jax.process_count = lambda: int(sys.argv[1])
    checkpointed = checkpoint_transition(mapped)
finally:
    jax.process_count = count

def loss(transition, value):
    result, _ = jax.lax.scan(lambda current, _: (transition(current), None), value, None, length=2)
    return jnp.sum(jnp.arange(1, 9).reshape(2, 4) * result)

point = jnp.linspace(0.1, 0.8, 8).reshape(2, 4)
all_saved = jax.checkpoint(mapped, policy=jax.checkpoint_policies.everything_saveable)
functions = [jax.jit(jax.grad(partial(loss, f))) for f in (mapped, jax.checkpoint(mapped), checkpointed, all_saved)]
gradients = [np.asarray(function(point)) for function in functions]
for gradient in gradients[1:]:
    error = np.max(np.abs(gradient - gradients[0]))
    assert error < 2e-12, f"ERROR checkpoint gradient max absolute error={error}"
direction = jnp.cos(point)
delta = 1e-4
finite_difference = (loss(mapped, point + delta * direction) - loss(mapped, point - delta * direction)) / (2 * delta)
error = abs(float(jnp.sum(gradients[2] * direction) - finite_difference))
assert error < 2e-7, f"ERROR checkpoint finite-difference absolute error={error}"
programs = [function.lower(point).compile().as_text() for function in functions]
counts = [program.count("collective-permute(") for program in programs]
assert counts[1] > counts[0], f"ERROR fixture has no replayed exchanges: {counts}"
expected = counts[0] if int(sys.argv[1]) > 1 else counts[1]
assert counts[2] == expected, f"ERROR checkpoint exchanges direct/plain/selected={counts}"
recomputed_math = [any("rematted_computation" in line and ("sine(" in line or "cosine(" in line) for line in program.splitlines()) for program in programs]
assert recomputed_math == [False, True, True, False], f"ERROR checkpoint arithmetic recomputation direct/plain/selected/all-saved={recomputed_math}"
print(f"checkpoint process_count={sys.argv[1]}: derivatives pass; exchanges={counts}")
"""


@pytest.mark.parametrize("process_count", [1, 2])
def test_checkpoint_preserves_derivatives_and_avoids_distributed_exchange_replay(
    tmp_path: Path, process_count: int
) -> None:
    """Use real two-device derivatives and compiled communication counts."""
    script = tmp_path / "checkpoint_probe.py"
    script.write_text(_PROBE)
    project = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        PYTHONPATH=str(project),
        JAX_PLATFORMS="cpu",
        JAX_NUM_CPU_DEVICES="2",
    )
    result = subprocess.run(
        [sys.executable, str(script), str(process_count)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR checkpoint probe: {result.stdout[-500:]} {result.stderr[-1500:]}"
    )
