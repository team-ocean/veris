"""One shared initializer owns future providers and all maintained adapters."""

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from tests.lifecycle_sharding_probe import _custom
from veris import Experiment, Grid, Simulation, initialize_experiment
from veris.grid.ocean import initialize_from_ocean
from veris.io import OutputSettings
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import Parameter, State
from veris.setups import island, presets, run_growth


def _scenario(values: Mapping[str, Any], dtype: str) -> dict[str, Any]:
    return {"column_ice": float(values.get("column_ice", 0.6))}


def _fields(
    state: State,
    configuration: Configuration,
    constants: PhysicalConstants,
    scenario: Mapping[str, Any],
    global_shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    *,
    regional: bool,
) -> dict[str, Any]:
    return {
        "hIceMean": jnp.full_like(state.hIceMean, scenario["column_ice"]),
        "hSnowMean": jnp.zeros_like(state.hSnowMean),
        "Area": jnp.full_like(state.Area, 0.8),
    }


def test_registered_provider_initializes_and_runs_without_core_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future provider must initialize derived fields through the common path."""
    presets.get_preset("growth")
    monkeypatch.setattr(presets, "_PRESETS", dict(presets._PRESETS))
    presets.register_preset(
        presets.SetupPreset(
            name="future_column",
            physics="thermodynamics",
            grid=(4, 6),
            scenario_registry={"column_ice": Parameter(0.6, float, "Initial ice", "m")},
            scenario_factory=_scenario,
            field_factory=_fields,
            numerical_defaults={"deltatTherm": 60.0, "nITC": 1},
            initial_defaults=lambda values, constants: {
                "hIceMean": values["column_ice"],
                "hSnowMean": 0.0,
            },
        )
    )
    definition = Experiment.builtin(
        "future_column", constants={"rhoIce": 900.0, "hIce_reg": 0.01}, steps=1
    )
    with Simulation.initialize(definition) as model:
        np.testing.assert_array_equal(model.state.hIceMean, 0.6)
        np.testing.assert_array_equal(model.state.SeaIceLoad, 540.0)
        np.testing.assert_allclose(
            model.state.recip_hIceMean, 1 / np.sqrt(0.37), rtol=1e-14
        )
        result = model.run()
        assert result.completed_steps == 1 and result.elapsed.total_seconds() == 60.0
        assert all(
            np.isfinite(getattr(result.state, item.name)).all()
            for item in fields(State)
        )


@pytest.mark.parametrize(
    "scenario,message",
    [
        ({"hIceMean": -1.0}, "hIceMean"),
        ({"hSnowMean": -0.1}, "hSnowMean"),
        ({"Area": 1.1}, "Area"),
        ({"TSurf": 0.0}, "TSurf"),
        ({"wSpeed": -1.0}, "wSpeed"),
        ({"ocSalt": -1.0}, "ocSalt"),
        ({"steps": -1}, "steps"),
        ({"steps": 2.5}, "steps"),
        ({"steps": True}, "steps"),
    ],
)
def test_growth_adapter_uses_shared_physical_validation(
    scenario: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_growth.initialize(scenario_overrides=scenario)


def test_legacy_initialization_does_not_require_a_runnable_host_clock() -> None:
    """Pure kernels may initialize sub-microsecond clocks unavailable to the facade."""
    controls = {"deltatDyn": 1e-9, "deltatTherm": 1e-9}
    state, conf, _ = island.initialize(4, 6, settings_overrides=controls)
    assert conf.deltatDyn == conf.deltatTherm == 1e-9
    assert np.isfinite(state.hIceMean).all()
    with pytest.raises(ValueError, match="microseconds"):
        Experiment.builtin("island", grid=Grid(4, 6), settings=controls)


def test_ocean_adapter_and_experiment_reject_the_same_invalid_mask() -> None:
    geometry = _custom(Grid(12, 16), True, OutputSettings(enabled=False)).geometry
    assert geometry is not None
    bad = replace(geometry, maskT=np.full((12, 16, 1), 1.2))
    with pytest.raises(ValueError, match="maskT"):
        initialize_from_ocean(bad)
    with pytest.raises(ValueError, match="maskT"):
        initialize_experiment(Experiment(grid=Grid(12, 16), geometry=bad))


@pytest.mark.parametrize(
    "name,value",
    [
        ("hIceMean", -1.0),
        ("Area", 1.1),
        ("TSurf", 0.0),
        ("theta", 0.0),
        ("wSpeed", -1.0),
        ("ocSalt", -1.0),
        ("Qnet", 1e40),
    ],
)
def test_ocean_adapter_rejects_invalid_fields_before_allocation(
    monkeypatch: pytest.MonkeyPatch, name: str, value: float
) -> None:
    from veris.grid import ocean

    geometry = _custom(Grid(12, 16), True, OutputSettings(enabled=False)).geometry
    assert geometry is not None

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR ocean adapter allocated before validating physical fields")

    monkeypatch.setattr(ocean.npx, "asarray", unexpected)
    with pytest.raises(ValueError, match=name):
        initialize_from_ocean(
            geometry,
            dtype="float32",
            state_overrides={name: np.full((12, 16), value)},
        )


@pytest.mark.parametrize(
    "name,value", [("dxt", 1e-46), ("dxt", 1e-39), ("area_t", 1e38), ("ht", 1e40)]
)
def test_ocean_adapter_rejects_unrepresentable_geometry_before_allocation(
    monkeypatch: pytest.MonkeyPatch, name: str, value: float
) -> None:
    from veris.grid import ocean

    geometry = _custom(Grid(12, 16), True, OutputSettings(enabled=False)).geometry
    assert geometry is not None
    bad = replace(geometry, **{name: np.full(np.shape(getattr(geometry, name)), value)})

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR ocean adapter allocated before validating precision")

    monkeypatch.setattr(ocean.npx, "asarray", unexpected)
    with pytest.raises(ValueError, match=name):
        initialize_from_ocean(bad, dtype="float32")
