"""Host execution resources shared by Python and command-line experiments.

No physical equations live here. Global meshes use every selected backend
device, with hybrid construction for multiple slices/process granules. Explicit
bootstrap inputs override launch environments. Services acquired here are closed
once; caller-owned services and meshes remain borrowed. Imports change no JAX
policy and enumerate no devices.
"""

import os
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from veris.execution.experiment_initialization import _validate_inputs
from veris.model.experiment import DistributedExecution, Experiment


def _integer(value: str, name: str) -> int:
    """Decode launch integers with field-specific diagnostics."""
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error


def distributed_options(
    environment: Mapping[str, str],
    backend: str,
    explicit: DistributedExecution | None = None,
) -> dict[str, Any] | None:
    """Resolve validated explicit controls or the existing JAX/Slurm environment."""
    if explicit is not None:
        options: dict[str, Any] = (
            {"cluster_detection_method": explicit.cluster_detection_method}
            if explicit.cluster_detection_method is not None
            else {
                "coordinator_address": explicit.coordinator_address,
                "num_processes": explicit.num_processes,
                "process_id": explicit.process_id,
            }
        )
        if explicit.local_device_ids is not None:
            options["local_device_ids"] = list(explicit.local_device_ids)
        elif backend == "cpu" or explicit.cluster_detection_method is None:
            options["local_device_ids"] = [0]
        return options
    names = ("JAX_COORDINATOR_ADDRESS", "JAX_NUM_PROCESSES", "JAX_PROCESS_ID")
    present = [name in environment for name in names]
    if any(present):
        if not all(present):
            raise ValueError(
                "explicit coordinator/count/process variables must be set together"
            )
        ids = (0,)
        if backend == "gpu":
            ids = tuple(
                _integer(value, "local device ID")
                for value in environment.get(
                    "JAX_LOCAL_DEVICE_IDS", environment.get("SLURM_LOCALID", "0")
                ).split(",")
            )
        control = DistributedExecution(
            coordinator_address=environment[names[0]],
            num_processes=_integer(environment[names[1]], "num_processes"),
            process_id=_integer(environment[names[2]], "process_id"),
            local_device_ids=ids,
        )
        return distributed_options({}, backend, control)
    if "SLURM_PROCID" in environment:
        count = _integer(environment.get("SLURM_NTASKS", "1"), "process count")
        rank = _integer(environment["SLURM_PROCID"], "process rank")
        if count < 1 or not 0 <= rank < count:
            raise ValueError(
                "Slurm process rank must lie within a positive process count"
            )
        if count > 1:
            return distributed_options(
                {}, backend, DistributedExecution(cluster_detection_method="slurm")
            )
    return None


def create_mesh(partitions: tuple[int, int] | None, backend: str) -> Mesh:
    """Create an all-device x/y mesh, including supported multi-slice topologies."""
    devices = jax.devices(backend)
    shape = (1, len(devices)) if partitions is None else partitions
    if (
        len(shape) != 2
        or any(type(size) is not int or size < 1 for size in shape)
        or shape[0] * shape[1] != len(devices)
    ):
        raise ValueError(
            f"mesh {shape} must contain all {len(devices)} available {backend} devices"
        )
    process_granules = not all(hasattr(device, "slice_index") for device in devices)
    attribute = "process_index" if process_granules else "slice_index"
    counts = Counter(getattr(device, attribute) for device in devices)
    if len(counts) == 1:
        return jax.make_mesh(shape, ("x", "y"), devices=devices)
    if len(set(counts.values())) != 1:
        raise ValueError("multi-slice meshes require equally sized device granules")
    granules = len(counts)
    outer = next(
        (
            (x, granules // x)
            for x in range(1, shape[0] + 1)
            if granules % x == 0
            and shape[0] % x == 0
            and shape[1] % (granules // x) == 0
        ),
        None,
    )
    if outer is None:
        raise ValueError(f"mesh {shape} cannot factor {granules} device granules")
    inner = tuple(size // divisor for size, divisor in zip(shape, outer, strict=True))
    arrays = mesh_utils.create_hybrid_device_mesh(
        inner, outer, devices=devices, process_is_granule=process_granules
    )
    return Mesh(arrays, ("x", "y"))


def _validate_mesh(experiment: Experiment, mesh: Mesh) -> None:
    """Check borrowed resources before JAX policy changes or device enumeration."""
    if not isinstance(mesh, Mesh) or set(mesh.axis_names) != {"x", "y"}:
        raise ValueError("mesh must be a JAX Mesh with x and y axes")
    execution = experiment.execution
    shape = mesh.shape["x"], mesh.shape["y"]
    if not execution.all_devices and shape != execution.partitions:
        raise ValueError("mesh shape must match Execution partitions")
    if shape == (1, 1) and not execution.all_devices:
        raise ValueError("serial execution does not require an explicit mesh")
    if execution.backend is not None:
        platform = "cpu" if execution.backend == "cpu" else "gpu"
        if any(device.platform != platform for device in mesh.devices.flat):
            raise ValueError("mesh devices do not match Execution backend")
    experiment.grid.local_shape(shape)


def _execution_metadata(backend: str, partitions: tuple[int, int]) -> Mapping[str, Any]:
    """Record actual backend/topology independently of authored launch controls."""
    return MappingProxyType(
        {
            "backend": backend,
            "partitions": partitions,
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
        }
    )


@dataclass(frozen=True)
class RuntimeResources:
    """Resolved numerical definition, borrowed mesh and immutable launch metadata."""

    experiment: Experiment
    mesh: Mesh | None
    device: Any
    owns_distributed: bool
    metadata: Mapping[str, Any]


@contextmanager
def execution_resources(
    experiment: Experiment,
    *,
    mesh: Mesh | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[RuntimeResources]:
    """Preflight inputs, acquire requested resources and close owned services.

    Distributed bootstrap precedes device enumeration. Explicit CPU launches
    retain one local device per rank and Gloo collectives. Precision/backend
    policy is applied only during this explicit launch operation. A supplied
    mesh with distributed launch controls requires a preinitialized service.
    """
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    _validate_inputs(experiment)
    execution = experiment.execution
    options = distributed_options(
        os.environ if environment is None else environment,
        execution.backend or "cpu",
        execution.distributed,
    )
    if options is not None:
        if execution.backend is None:
            raise ValueError("distributed execution requires an explicit backend")
        if not execution.all_devices and execution.partitions == (1, 1):
            raise ValueError("distributed execution requires a partitioned mesh")
    if mesh is not None:
        _validate_mesh(experiment, mesh)
    borrowed = jax.distributed.is_initialized()
    if mesh is not None and options is not None and not borrowed:
        raise ValueError(
            "a supplied mesh requires a preinitialized distributed service"
        )
    jax.config.update("jax_enable_x64", True)
    if execution.backend is not None:
        jax.config.update(
            "jax_platforms", "cuda" if execution.backend == "gpu" else "cpu"
        )
    owned = False
    try:
        if options is not None and not borrowed:
            if execution.backend == "cpu":
                jax.config.update("jax_num_cpu_devices", 1)
                jax.config.update("jax_cpu_collectives_implementation", "gloo")
            owned = True
            jax.distributed.initialize(**options)
        if (
            borrowed
            and options is not None
            and (
                (
                    "num_processes" in options
                    and options["num_processes"] != jax.process_count()
                )
                or (
                    "process_id" in options
                    and options["process_id"] != jax.process_index()
                )
            )
        ):
            raise ValueError("distributed controls do not match the borrowed service")
        backend = execution.backend
        if backend is None:
            if mesh is not None:
                backend = mesh.devices.flat[0].platform
            else:
                default = jax.config.jax_default_device
                backend = getattr(default, "platform", None) or jax.default_backend()
        selected = mesh
        if selected is None and (
            execution.all_devices or execution.partitions != (1, 1)
        ):
            selected = create_mesh(
                None if execution.all_devices else execution.partitions, backend
            )
        shape = (
            (1, 1) if selected is None else (selected.shape["x"], selected.shape["y"])
        )
        experiment.grid.local_shape(shape)
        resolved = experiment
        if execution.all_devices:
            resolved = experiment.with_overrides(
                {"execution": {"all_devices": False, "partitions": shape}}
            )
        count = jax.process_count()
        if count > 1 and shape == (1, 1):
            raise ValueError("distributed execution requires a partitioned mesh")
        device = (
            jax.devices(backend)[0] if selected is None else selected.local_devices[0]
        )
        metadata = _execution_metadata(backend, shape)
        yield RuntimeResources(
            resolved, None if shape == (1, 1) else selected, device, owned, metadata
        )
    finally:
        if owned:
            jax.distributed.shutdown()


__all__ = [
    "RuntimeResources",
    "create_mesh",
    "distributed_options",
    "execution_resources",
]
