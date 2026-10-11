"""Public definition contracts; numerical registry tests own scalar equations."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

import veris
from veris._typing import OceanGeometry
from veris.configuration import SETTINGS, Configuration
from veris.io import OutputSettings, Stream
from veris.physical_constants import PhysicalConstants


def test_global_grid_adapts_to_local_extents_and_packed_partition_halos() -> None:
    grid = veris.Grid(12, 8)
    assert grid.local_shape() == (12, 8)
    assert grid.storage_shape() == (16, 12)
    assert grid.local_shape((3, 2)) == (4, 4)
    assert grid.storage_shape((3, 2)) == (24, 16)
    assert veris.Grid().nx == SETTINGS["nx"].default
    assert veris.Grid().ny == SETTINGS["ny"].default
    with pytest.raises(FrozenInstanceError):
        grid.nx = 4  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "dimensions, partitions, message",
    [
        ((True, 8), (1, 1), "nx"),
        ((1, 8), (1, 1), "nx"),
        ((8, 2.5), (1, 1), "ny"),
        ((12, 8), (5, 1), "divis"),
        ((12, 8), (12, 1), "two"),
        ((12, 8), (0, 1), "partition"),
        ((12, 8), (True, 1), "partition"),
        ((12, 8), (1,), "partition"),
    ],
)
def test_invalid_grids_fail_before_allocation(
    dimensions: tuple[Any, Any], partitions: tuple[Any, ...], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        veris.Grid(*dimensions).local_shape(partitions)


def test_definition_resolves_registry_objects_and_separates_resources() -> None:
    definition = veris.Experiment(
        grid=veris.Grid(12, 8),
        settings={"dtype": "float32", "deltatDyn": 300, "deltatTherm": 300},
        constants={"rhoIce": 910},
        execution=veris.Execution(backend="cpu", partitions=(3, 2)),
        steps=4,
    )
    assert isinstance(definition.configuration, Configuration)
    assert isinstance(definition.physical_constants, PhysicalConstants)
    assert (definition.configuration.nx, definition.configuration.ny) == (4, 4)
    assert definition.configuration.use_sharding
    assert (
        definition.configuration.dtype
        == definition.physical_constants.dtype
        == "float32"
    )
    assert definition.configuration.nEVPsteps == SETTINGS["nEVPsteps"].default
    assert definition.physical_constants.rhoIce == 910
    assert (
        definition.physical_constants.rhoSnow
        == PhysicalConstants(dtype="float32").rhoSnow
    )
    assert definition.step_duration == timedelta(seconds=300)
    assert not veris.Experiment().configuration.use_sharding
    assert replace(definition, execution=veris.Execution()).configuration.nx == 12


def test_initial_fields_are_validated_and_frozen() -> None:
    fields = {"hIceMean": np.ones((2, 4)), "Area": 0.8}
    definition = veris.Experiment(grid=veris.Grid(2, 4), initial_conditions=fields)
    fields["hIceMean"][:] = 9
    fields["Area"] = -1
    np.testing.assert_array_equal(definition.initial_conditions["hIceMean"], 1)
    assert definition.initial_conditions["Area"] == 0.8
    with pytest.raises((ValueError, AttributeError), match="immutable"):
        definition.initial_conditions["hIceMean"].shape = (4, 2)
    with pytest.raises(TypeError):
        definition.settings["nEVPsteps"] = 5  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"settings": {"nx": 12}}, "Grid"),
        ({"settings": {"use_sharding": True}}, "Execution"),
        ({"settings": {"unknown": 1}}, "unknown"),
        ({"constants": {"unknown": 1}}, "unknown"),
        (
            {"settings": {"dtype": "float32"}, "constants": {"dtype": "float64"}},
            "dtype",
        ),
        ({"settings": {"deltatDyn": 300, "deltatTherm": 600}}, "coupled.*timestep"),
        ({"physics": "ocean"}, "physics"),
        ({"steps": True}, "steps"),
        ({"steps": -1}, "steps"),
        ({"initial_conditions": {"SeaIceLoad": 1}}, "SeaIceLoad"),
        ({"initial_conditions": {"hIceMean": -1}}, "hIceMean.*nonnegative"),
        ({"initial_conditions": {"Area": 1.1}}, "Area"),
        ({"initial_conditions": {"TSurf": 0}}, "TSurf.*positive"),
        ({"initial_conditions": {"hSnowMean": np.inf}}, "hSnowMean.*finite"),
        ({"initial_conditions": {"hIceMean": np.ones((6, 8))}}, "hIceMean.*shape"),
    ],
)
def test_invalid_experiment_has_informative_error(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        veris.Experiment(**kwargs)


def test_thermodynamic_and_dynamic_clocks_use_their_active_timestep() -> None:
    settings = {"deltatDyn": 0.5, "deltatTherm": 1.25}
    assert veris.Experiment(
        physics="dynamics", settings=settings
    ).step_duration == timedelta(microseconds=500000)
    assert veris.Experiment(
        physics="thermodynamics", settings=settings
    ).step_duration == timedelta(microseconds=1250000)
    with pytest.raises(ValueError, match="microsecond"):
        veris.Experiment(physics="dynamics", settings={"deltatDyn": 1e-8})


def test_definition_accepts_host_output_separately_from_physics() -> None:
    output = OutputSettings(streams=(Stream(sampling_interval=timedelta(seconds=300)),))
    definition = veris.Experiment(output=output)
    assert definition.output is output
    assert definition.configuration == veris.Experiment().configuration


def test_definition_rejects_duration_beyond_supplied_forcing() -> None:
    with pytest.raises(ValueError, match="forcing.*length"):
        veris.Experiment(forcing=veris.Forcing(series={"Qnet": [1]}), steps=2)


@pytest.mark.parametrize("kwargs", [{"backend": "tpu"}, {"partitions": (0, 1)}])
def test_invalid_execution_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError), match="backend|partition"):
        veris.Execution(**kwargs)


def test_definition_copies_custom_geometry_without_mutating_ocean_input() -> None:
    horizontal = np.ones((2, 4))
    values: dict[str, Any] = {
        "maskT": horizontal[..., None],
        "maskU": horizontal[..., None],
        "maskV": horizontal[..., None],
        "ht": -100 * horizontal,
        "coriolis_t": 1e-4 * horizontal,
        "dxt": np.ones(2),
        "dxu": np.ones(2),
        "dyt": np.ones(4),
        "dyu": np.ones(4),
        "area_t": horizontal,
        "area_u": horizontal,
        "area_v": horizontal,
    }
    geometry = OceanGeometry(**values)
    definition = veris.Experiment(grid=veris.Grid(2, 4), geometry=geometry)
    horizontal[:] = 0
    assert definition.geometry is not None
    for item in fields(geometry):
        saved = np.asarray(getattr(definition.geometry, item.name))
        with pytest.raises(ValueError):
            saved.setflags(write=True)
    np.testing.assert_array_equal(definition.geometry.maskT, 1)
    np.testing.assert_array_equal(geometry.maskT, 0)
    with pytest.raises((ValueError, AttributeError), match="immutable"):
        # Avoid a static deprecated-shape warning for this intentional mutation.
        setattr(definition.geometry.maskT, "shape", (4, 2, 1))  # noqa: B010


def test_clock_overflow_is_rejected_with_host_clock_context() -> None:
    with pytest.raises(ValueError, match="clock"):
        veris.Experiment(physics="dynamics", settings={"deltatDyn": 1e308})


def test_public_definition_import_does_not_initialize_backend_or_change_precision() -> (
    None
):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import jax
from jax._src import xla_bridge
before = jax.config.jax_enable_x64
import veris
definition = veris.Experiment()
assert jax.config.jax_enable_x64 == before
assert not xla_bridge.backends_are_initialized()
assert definition.configuration.dtype == 'float64'
""",
        ],
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "false"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
