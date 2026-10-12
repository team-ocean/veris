"""A coordinated pytest session isolates worker CPUs and their fresh JAX caches."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import conftest
import pytest


@pytest.mark.parametrize("index", [0, 1, 2])
def test_worker_affinity_partitions_only_permitted_cpus(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    """Workers receive disjoint strided groups before initializing their backend."""
    assigned: list[set[int]] = []
    permitted = {0, 2, 3, 8, 10, 11}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: permitted, raising=False)
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: assigned.append(cpus), raising=False
    )
    config: Any = SimpleNamespace(
        workerinput={"workerid": f"gw{index}", "workercount": 3},
        getoption=lambda name: True,
    )
    conftest._partition_worker_cpus(config)
    assert assigned == [set(sorted(permitted)[index::3])]


def test_worker_affinity_preserves_unrequested_and_controller_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serial callers and controllers keep their existing process placement."""

    def unexpected(*args: Any) -> None:
        pytest.fail("ERROR affinity changed outside an explicitly partitioned worker")

    monkeypatch.setattr(os, "sched_setaffinity", unexpected, raising=False)
    unrequested: Any = SimpleNamespace(getoption=lambda name: False)
    controller: Any = SimpleNamespace(getoption=lambda name: True)
    conftest._partition_worker_cpus(unrequested)
    conftest._partition_worker_cpus(controller)


@pytest.mark.parametrize(
    "index,count,allowed", [(0, 2, {1}), (3, 3, {0, 1, 2, 3, 4, 5})]
)
def test_worker_affinity_rejects_empty_or_restarted_groups(
    monkeypatch: pytest.MonkeyPatch, index: int, count: int, allowed: set[int]
) -> None:
    """Invalid placement fails explicitly rather than overlapping another worker."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: allowed, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None, raising=False)
    config: Any = SimpleNamespace(
        workerinput={"workerid": f"gw{index}", "workercount": count},
        getoption=lambda name: True,
    )
    with pytest.raises(pytest.UsageError):
        conftest._partition_worker_cpus(config)


def test_worker_affinity_reports_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly requested unavailable placement option has a clear host error."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    config: Any = SimpleNamespace(
        workerinput={"workerid": "gw0", "workercount": 1},
        getoption=lambda name: True,
    )
    with pytest.raises(pytest.UsageError, match="CPU affinity support"):
        conftest._partition_worker_cpus(config)


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"),
    reason="native CPU affinity is unavailable",
)
def test_worker_processes_clean_private_caches_and_inherit_affinity() -> None:
    """Real child numerics exercise placement and independent cache ownership."""
    root = Path(__file__).resolve().parents[1]
    script = r"""
import json, os, sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from conftest import _partition_worker_cpus, _temporary_compilation_cache

os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
os.environ.pop("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", None)
jax.config.update("jax_compilation_cache_dir", None)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
index = int(sys.argv[1])
count = min(2, len(os.sched_getaffinity(0)))
config = SimpleNamespace(workerinput={"workerid": f"gw{index % count}", "workercount": count}, getoption=lambda name: True)
_partition_worker_cpus(config)
with _temporary_compilation_cache():
    result = jax.jit(lambda x: x * x + 1)(jnp.asarray(index + 2))
    assert int(result) == (index + 2) ** 2 + 1
    record = {"cache": os.environ["JAX_COMPILATION_CACHE_DIR"], "threshold": jax.config.jax_persistent_cache_min_compile_time_secs, "affinity": sorted(os.sched_getaffinity(0))}
print(json.dumps(record))
"""
    records = []
    for index in range(2):
        result = subprocess.run(
            [sys.executable, "-c", script, str(index)],
            env=dict(
                os.environ,
                JAX_PLATFORMS="cpu",
                PYTHONPATH=os.pathsep.join((str(root / "tests"), str(root))),
            ),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"ERROR worker runtime: {result.stderr[-1500:]}"
        records.append(json.loads(result.stdout))
    assert records[0]["cache"] != records[1]["cache"]
    assert all(not Path(record["cache"]).exists() for record in records)
    assert [record["threshold"] for record in records] == [0.0, 0.0]
    affinities = [set(record["affinity"]) for record in records]
    if len(os.sched_getaffinity(0)) == 1:
        assert affinities[0] == affinities[1]
    else:
        assert not affinities[0] & affinities[1]
    assert affinities[0] | affinities[1] == os.sched_getaffinity(0)
