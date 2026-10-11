"""A fresh two-CPU process exercises direct advance with spatial forcing."""

import os
import subprocess
import sys
from pathlib import Path


def test_pure_advance_two_cpu_orientations(tmp_path: Path) -> None:
    """Check packed State, physical forcing, closed walls and real distributed AD."""
    # Locate the project while this test is a draft and after moving into tests/.
    root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "setup.cfg").is_file() and (parent / "DESIGN.md").is_file()
    )
    probe = Path(__file__).with_name("model_physics_sharding_probe.py")
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_NUM_CPU_DEVICES="2",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
        PYTHONPATH=str(root),
    )
    log = tmp_path / "pure-advance-two-cpu.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(probe)],
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=240,
            check=False,
        )
    output = log.read_text(encoding="utf-8")
    assert result.returncode == 0, (
        f"ERROR direct advance two-CPU probe: {output[-2500:]}"
    )
    assert "pure advance two-CPU checks passed" in output
