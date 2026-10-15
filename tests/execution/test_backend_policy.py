"""A CPU request must preserve other caller-configured backend availability."""

import os
import subprocess
import sys


def test_serial_cpu_launch_preserves_caller_platform_configuration() -> None:
    script = """
import jax
from veris import Execution, Experiment, run_experiment
before = jax.config.jax_platforms
assert before is None
run_experiment(Experiment.builtin("growth", steps=0, execution=Execution(backend="cpu")))
assert jax.config.jax_platforms == before, "ERROR CPU launch narrowed caller platform configuration"
"""
    environment = dict(os.environ)
    environment.pop("JAX_PLATFORMS", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1800:]
