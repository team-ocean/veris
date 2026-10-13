"""Reproduce final-review gaps in cold topology, precision and control ownership."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import jax
import numpy as np
import pytest

from veris import (
    DistributedExecution,
    Execution,
    Experiment,
    Grid,
    initialize_experiment,
)
from veris.execution.runtime import execution_resources
from veris.model.physical_constants import PhysicalConstants
from veris.setups import island, presets


@pytest.mark.parametrize("processes,partitions", [(3, (1, 2)), (2, (1, 4))])
def test_owned_cpu_topology_rejects_impossible_counts_before_effects(
    monkeypatch: pytest.MonkeyPatch,
    processes: int,
    partitions: tuple[int, int],
) -> None:
    model = Experiment.builtin(
        "growth",
        grid=Grid(4, 8),
        execution=Execution(
            backend="cpu",
            partitions=partitions,
            distributed=DistributedExecution(
                coordinator_address="localhost:12345",
                num_processes=processes,
                process_id=0,
            ),
        ),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR impossible CPU topology reached launch effects")

    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(jax.config, "update", forbidden)
    monkeypatch.setattr(jax.distributed, "initialize", forbidden)
    with (
        pytest.raises(ValueError, match="process|rank|CPU"),
        execution_resources(model, environment={}),
    ):
        pytest.fail("ERROR impossible CPU topology accepted")


@pytest.mark.parametrize("callable_defaults", [False, True])
def test_preset_precision_defaults_control_final_scenario_resolution(
    monkeypatch: pytest.MonkeyPatch, callable_defaults: bool
) -> None:
    base = presets.get_preset("growth")
    assert isinstance(base.numerical_defaults, Mapping)
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    numerical = {**base.numerical_defaults, "dtype": "float32"}
    provider = replace(
        base,
        name="review_precision",
        numerical_defaults=(lambda values: numerical)
        if callable_defaults
        else numerical,
        profiles={"high": {"settings": {"dtype": "float64"}}},
    )
    presets.register_preset(provider)
    implicit = Experiment.builtin(provider.name)
    explicit = implicit.with_overrides({"settings": {"dtype": "float32"}})
    assert implicit.configuration.dtype == explicit.configuration.dtype == "float32"
    assert implicit.resolved_scenario == explicit.resolved_scenario
    assert implicit.resolved_scenario["hIceMean"] == float(np.float32(1.3))
    assert (
        Experiment.builtin(provider.name, profile="high").resolved_scenario["hIceMean"]
        == 1.3
    )
    assert (
        implicit.with_overrides({"settings": {"dtype": "float64"}}).resolved_scenario[
            "hIceMean"
        ]
        == 1.3
    )


def test_authored_scenario_arrays_are_owned_and_exposed_readonly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = presets.get_preset("growth")
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))

    def resolve(values: Any, dtype: str) -> Any:
        return base.scenario_factory(
            {**values, "hIceMean": float(np.asarray(values.get("hIceMean", 0.6)))},
            dtype,
        )

    provider = replace(base, name="review_array", scenario_factory=resolve)
    presets.register_preset(provider)
    value = np.array(0.6)
    model = Experiment.builtin(provider.name, scenario={"hIceMean": value})
    value[...] = 0.9
    assert float(model.scenario["hIceMean"]) == 0.6
    assert model.with_overrides({}).resolved_scenario == model.resolved_scenario
    with pytest.raises(ValueError, match="read-only|readonly"):
        model.scenario["hIceMean"][...] = 0.8


def test_authored_constant_sequences_are_owned_across_noop_resolution() -> None:
    original = PhysicalConstants().longwaveCloudCoefficients
    values = list(original)
    model = Experiment.builtin(
        "growth", constants={"longwaveCloudCoefficients": values}
    )
    values[0] += 1.0
    assert tuple(model.constants["longwaveCloudCoefficients"]) == original
    assert model.with_overrides({}).physical_constants == model.physical_constants


@pytest.mark.parametrize("legacy", [False, True])
def test_island_salinity_alias_obeys_shared_physical_bound_before_allocation(
    monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    from veris.execution import initialization

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR negative island salinity reached allocation")

    monkeypatch.setattr(initialization, "initialize", forbidden)
    with pytest.raises(ValueError, match="ocSalt|salinity|saltOcn"):
        if legacy:
            island.initialize(4, 4, scenario_overrides={"saltOcn_ref": -1.0})
        else:
            initialize_experiment(
                Experiment.builtin("island", scenario={"saltOcn_ref": -1.0})
            )


@pytest.mark.parametrize("cluster", [False, True])
def test_owned_cpu_known_topology_is_cold_for_cluster_and_all_device_requests(
    monkeypatch: pytest.MonkeyPatch, cluster: bool
) -> None:
    control = (
        DistributedExecution(cluster_detection_method="slurm")
        if cluster
        else DistributedExecution(
            coordinator_address="localhost:12345", num_processes=3, process_id=0
        )
    )
    model = Experiment.builtin(
        "growth",
        grid=Grid(4, 8),
        execution=Execution(
            backend="cpu",
            partitions=(1, 2) if cluster else (1, 1),
            all_devices=not cluster,
            distributed=control,
        ),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR invalid known topology reached launch effects")

    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(jax.config, "update", forbidden)
    with (
        pytest.raises(ValueError, match="process|partition|divis"),
        execution_resources(model, environment={"SLURM_NTASKS": "3"}),
    ):
        pytest.fail("ERROR invalid known topology accepted")


def test_callable_precision_cycles_are_rejected_unless_explicitly_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = presets.get_preset("growth")
    assert isinstance(base.numerical_defaults, Mapping)
    numerical = dict(base.numerical_defaults)
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))

    def defaults(values: Any) -> Any:
        return {
            **numerical,
            "dtype": "float32" if float(values["hIceMean"]) == 1.3 else "float64",
        }

    provider = replace(base, name="review_cycle", numerical_defaults=defaults)
    presets.register_preset(provider)
    with pytest.raises(ValueError, match="dtype.*stable"):
        Experiment.builtin(provider.name)
    explicit = Experiment.builtin(provider.name, settings={"dtype": "float32"})
    assert explicit.configuration.dtype == "float32"
    assert explicit.resolved_scenario["hIceMean"] == float(np.float32(1.3))


def test_owned_gpu_mesh_cannot_have_fewer_devices_than_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Experiment.builtin(
        "growth",
        grid=Grid(4, 8),
        execution=Execution(
            backend="gpu",
            partitions=(1, 2),
            distributed=DistributedExecution(
                coordinator_address="localhost:12345",
                num_processes=3,
                process_id=0,
            ),
        ),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR impossible GPU topology reached launch effects")

    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(jax.config, "update", forbidden)
    with (
        pytest.raises(ValueError, match="device.*process"),
        execution_resources(model, environment={}),
    ):
        pytest.fail("ERROR impossible GPU topology accepted")
