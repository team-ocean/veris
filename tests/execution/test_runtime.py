"""Cold launch validation, distributed ownership and physical mesh construction."""

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

import veris


def _distributed(**options: Any) -> Any:
    assert hasattr(veris, "DistributedExecution"), "missing DistributedExecution"
    return veris.DistributedExecution(**options)


def _runtime() -> Any:
    return import_module("veris.execution.runtime")


@pytest.mark.parametrize(
    "options,message",
    [
        ({}, "coordinator|cluster|distributed"),
        ({"coordinator_address": "host:1234"}, "together"),
        (
            {"coordinator_address": "host:1234", "num_processes": 0, "process_id": 0},
            "process",
        ),
        (
            {"coordinator_address": "host:1234", "num_processes": 2, "process_id": 2},
            "process",
        ),
        (
            {
                "coordinator_address": "host:1234",
                "num_processes": True,
                "process_id": 0,
            },
            "process",
        ),
        (
            {"coordinator_address": "", "num_processes": 2, "process_id": 0},
            "coordinator",
        ),
        (
            {"coordinator_address": "host", "num_processes": 2, "process_id": 0},
            "coordinator",
        ),
        ({"cluster_detection_method": "unknown"}, "cluster"),
        ({"cluster_detection_method": "slurm", "local_device_ids": (-1,)}, "device"),
        ({"cluster_detection_method": "slurm", "local_device_ids": (0, 0)}, "device"),
        ({"cluster_detection_method": "slurm", "local_device_ids": ()}, "device"),
        ({"cluster_detection_method": "slurm", "local_device_ids": (True,)}, "device"),
    ],
)
def test_distributed_controls_validate_without_selecting_devices(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any], message: str
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("ERROR distributed definition selected devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    with pytest.raises((ValueError, TypeError), match=message):
        _distributed(**options)


def test_explicit_distributed_controls_override_environment_and_round_trip() -> None:
    runtime = _runtime()
    explicit = _distributed(
        coordinator_address="explicit:12345",
        num_processes=2,
        process_id=1,
        local_device_ids=(3,),
    )
    assert runtime.distributed_options(
        {"JAX_PROCESS_ID": "invalid"}, "gpu", explicit
    ) == {
        "coordinator_address": "explicit:12345",
        "num_processes": 2,
        "process_id": 1,
        "local_device_ids": [3],
    }
    definition = veris.Experiment.builtin(
        "dynamics",
        grid=veris.Grid(4, 6),
        execution=veris.Execution(
            backend="gpu", partitions=(1, 2), distributed=explicit
        ),
    )
    from veris.model.experiment_file import experiment_from_dict

    restored = experiment_from_dict(definition.to_dict())
    assert restored.execution == definition.execution
    updated = restored.with_overrides({"execution": {"distributed": {"process_id": 0}}})
    assert updated.execution.distributed is not None
    assert updated.execution.distributed.process_id == 0


@pytest.mark.parametrize(
    "environment,message",
    [
        ({"JAX_PROCESS_ID": "0"}, "together"),
        (
            {
                "JAX_COORDINATOR_ADDRESS": "host:1234",
                "JAX_NUM_PROCESSES": "2",
                "JAX_PROCESS_ID": "bad",
            },
            "process",
        ),
        (
            {
                "JAX_COORDINATOR_ADDRESS": "host:1234",
                "JAX_NUM_PROCESSES": "2",
                "JAX_PROCESS_ID": "-1",
            },
            "process",
        ),
        ({"SLURM_NTASKS": "2", "SLURM_PROCID": "2"}, "process|rank"),
    ],
)
def test_invalid_environment_is_rejected_on_the_host(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _runtime().distributed_options(environment, "cpu")


@pytest.mark.parametrize(
    "borrowed,failure",
    [(False, None), (False, "mesh"), (False, "body"), (True, None), (True, "body")],
)
def test_runtime_closes_owned_services_once_and_keeps_borrowed_services(
    monkeypatch: pytest.MonkeyPatch, borrowed: bool, failure: str | None
) -> None:
    runtime = _runtime()
    events = []
    initialized = {"value": borrowed}

    def initialize(**options: Any) -> None:
        assert options["process_id"] == 0
        events.append("initialize")
        initialized["value"] = True

    def shutdown() -> None:
        events.append("shutdown")
        initialized["value"] = False

    def mesh(*args: Any, **kwargs: Any) -> Any:
        events.append("mesh")
        if failure == "mesh":
            raise RuntimeError("mesh creation failed")
        return SimpleNamespace(
            shape={"x": 1, "y": 2},
            local_devices=jax.local_devices(backend="cpu"),
            devices=np.asarray([SimpleNamespace(platform="cpu")] * 2).reshape(1, 2),
        )

    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: initialized["value"])
    monkeypatch.setattr(jax.distributed, "initialize", initialize)
    monkeypatch.setattr(jax.distributed, "shutdown", shutdown)
    monkeypatch.setattr(jax.config, "update", lambda *args: None)
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(runtime, "create_mesh", mesh)
    definition = veris.Experiment.builtin(
        "growth",
        grid=veris.Grid(4, 4),
        execution=veris.Execution(
            backend="cpu",
            partitions=(1, 2),
            distributed=_distributed(
                coordinator_address="host:1234", num_processes=2, process_id=0
            ),
        ),
    )

    def launch() -> None:
        with runtime.execution_resources(definition, environment={}) as resources:
            assert resources.owns_distributed is (not borrowed)
            assert resources.metadata["process_count"] == 2
            if failure == "body":
                raise RuntimeError("run failed")

    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            launch()
    else:
        launch()
    assert events.count("initialize") == (0 if borrowed else 1)
    assert events.count("shutdown") == (0 if borrowed else 1)
    assert initialized["value"] is borrowed


def test_multi_slice_mesh_uses_hybrid_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    devices = [
        SimpleNamespace(slice_index=s, id=s, process_index=s, platform="cpu")
        for s in range(2)
    ]
    calls = []

    def hybrid(inner: Any, outer: Any, *, devices: Any, **kwargs: Any) -> Any:
        calls.append((tuple(inner), tuple(outer)))
        return np.asarray(devices).reshape(1, 2)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR multi-slice topology reached unsupported jax.make_mesh")

    from jax.experimental import mesh_utils

    monkeypatch.setattr(jax, "devices", lambda backend: devices)
    monkeypatch.setattr(jax, "make_mesh", forbidden)
    monkeypatch.setattr(mesh_utils, "create_hybrid_device_mesh", hybrid)
    monkeypatch.setattr(
        runtime,
        "Mesh",
        lambda arrays, names: SimpleNamespace(devices=arrays, axis_names=names),
    )
    mesh = runtime.create_mesh((1, 2), "cpu")
    assert mesh.devices.shape == (1, 2) and mesh.axis_names == ("x", "y")
    assert calls == [((1, 1), (1, 2))]


def test_runtime_checks_borrowed_mesh_before_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    mesh = jax.make_mesh((1, 1), ("x", "y"), devices=jax.devices("cpu")[:1])
    definition = veris.Experiment.builtin(
        "growth", execution=veris.Execution(backend="gpu")
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR invalid borrowed mesh selected devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    with (
        pytest.raises(ValueError, match="backend|partition|serial"),
        runtime.execution_resources(definition, mesh=mesh, environment={}),
    ):
        pytest.fail("ERROR invalid mesh was accepted")


def test_all_device_launch_profile_is_resolved_at_launch() -> None:
    assert hasattr(veris.Execution(), "all_devices"), (
        "missing all-device launch request"
    )
    definition = veris.Experiment.builtin("dynamics", profile="parallel")
    assert definition.execution.all_devices is True
    assert veris.Execution().partitions == (1, 1) and not veris.Execution().all_devices


def test_partial_bootstrap_failure_closes_owned_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    events = []
    initialized = {"value": False}

    def initialize(**options: Any) -> None:
        initialized["value"] = True
        raise RuntimeError("bootstrap failed")

    def shutdown() -> None:
        events.append("shutdown")
        initialized["value"] = False

    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: initialized["value"])
    monkeypatch.setattr(jax.distributed, "initialize", initialize)
    monkeypatch.setattr(jax.distributed, "shutdown", shutdown)
    monkeypatch.setattr(jax.config, "update", lambda *args: None)
    definition = veris.Experiment.builtin(
        "growth",
        grid=veris.Grid(4, 4),
        execution=veris.Execution(
            backend="cpu",
            partitions=(1, 2),
            distributed=_distributed(
                coordinator_address="host:1234", num_processes=2, process_id=0
            ),
        ),
    )
    with (
        pytest.raises(RuntimeError, match="bootstrap failed"),
        runtime.execution_resources(definition, environment={}),
    ):
        pytest.fail("ERROR failed bootstrap entered body")
    assert events == ["shutdown"] and initialized["value"] is False
