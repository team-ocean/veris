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


@pytest.mark.parametrize("allowed,workers", [(2, 1), (4, 1), (8, 2), (12, 3), (48, 3)])
def test_automatic_workers_reserve_four_permitted_cpus(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
    allowed: int,
    workers: int,
) -> None:
    """Avoid undersized collective pools and cap concurrent compiler memory."""
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(allowed)), raising=False
    )
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert conftest.pytest_xdist_auto_num_workers(pytestconfig) == workers


def test_automatic_workers_fall_back_without_cpu_affinity(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> None:
    """Non-Linux hosts still choose one worker when CPU availability is unknown."""
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert conftest.pytest_xdist_auto_num_workers(pytestconfig) == 1


@pytest.mark.parametrize("platforms", [None, "cuda,cpu", "unknown"])
def test_automatic_gpu_workers_share_one_device_process(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, platforms: str | None
) -> None:
    """GPU probes own their meshes; automatic CPU scheduling must not multiply them."""
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    if platforms is None:
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    else:
        monkeypatch.setenv("JAX_PLATFORMS", platforms)
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(48)), raising=False
    )
    assert conftest.pytest_xdist_auto_num_workers(pytestconfig) == 1


@pytest.mark.parametrize("requested", [0, 5])
def test_automatic_workers_honor_explicit_environment_count(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, requested: int
) -> None:
    """Explicit xdist policy, including disabling workers, remains authoritative."""
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", str(requested))
    assert conftest.pytest_xdist_auto_num_workers(pytestconfig) == requested


def test_automatic_workers_ignore_invalid_environment_count(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> None:
    """An invalid xdist override warns and retains safe automatic placement."""
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "not-a-count")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(
        os, "sched_getaffinity", lambda pid: set(range(4)), raising=False
    )
    with pytest.warns(UserWarning, match="PYTEST_XDIST_AUTO_NUM_WORKERS"):
        assert conftest.pytest_xdist_auto_num_workers(pytestconfig) == 1


def test_default_partitioning_is_optional_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> None:
    """The real registered options keep serial tests usable without affinity."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    parser = pytest.Parser(_ispytest=True)
    conftest.pytest_addoption(parser)
    options = parser.parse_known_args(pytestconfig.getini("addopts"))
    config: Any = SimpleNamespace(
        workerinput={"workerid": "gw0", "workercount": 1},
        getoption=lambda name: options.partition_cpus,
    )
    conftest._partition_worker_cpus(config)
    options = parser.parse(["--partition-cpus"])
    with pytest.raises(pytest.UsageError, match="CPU affinity support"):
        conftest._partition_worker_cpus(config)
