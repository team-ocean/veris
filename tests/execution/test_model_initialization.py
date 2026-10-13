"""Runnable public initialization adapts physical inputs without changing presets.

Preset compatibility uses existing setup initializers as the numerical oracle.
Custom geometry and boundary expectations use independent physical-cell indices;
no production halo helper builds their expected values.
"""

import os
import subprocess
import sys
from dataclasses import fields, replace
from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

import veris
from veris.grid.geometry import OceanGeometry
from veris.model.types import State
from veris.setups import island, run_dyn, run_growth


@pytest.fixture
def physical_geometry() -> OceanGeometry:
    """Nonuniform 4 by 5 grid with dry, fractional and fully wet surface cells."""
    assert veris.OceanGeometry is OceanGeometry
    x, y = np.indices((4, 5))
    surface = np.ones((4, 5))
    surface[1, 2] = 0
    surface[2, 3] = 0.5
    west = surface * np.roll(surface, 1, axis=0)
    south = surface * np.roll(surface, 1, axis=1)
    return veris.OceanGeometry(
        maskT=np.stack((np.zeros_like(surface), surface), axis=-1),
        maskU=np.stack((np.zeros_like(surface), west), axis=-1),
        maskV=np.stack((np.zeros_like(surface), south), axis=-1),
        ht=-(80.0 + x + 2 * y),
        coriolis_t=(y - 2) * 1e-5,
        dxt=np.array([2000.0, 3000.0, 5000.0, 7000.0]),
        dxu=np.array([2500.0, 3500.0, 5500.0, 7500.0]),
        dyt=np.array([3000.0, 4000.0, 6000.0, 8000.0, 10000.0]),
        dyu=np.array([3500.0, 4500.0, 6500.0, 8500.0, 10500.0]),
        area_t=(10.0 + x + 2 * y) * 1e6,
        area_u=(20.0 + 2 * x + y) * 1e6,
        area_v=(30.0 + x + 3 * y) * 1e6,
    )


def _halo(physical: np.ndarray, cyclic_y: bool, boundary: str = "edge") -> np.ndarray:
    """Index physical cells independently and apply documented exterior values."""
    nx, ny = physical.shape
    x = (np.arange(nx + 4) - 2) % nx
    y = np.arange(ny + 4) - 2
    y = y % ny if cyclic_y else np.clip(y, 0, ny - 1)
    expected = physical[np.ix_(x, y)].copy()
    if not cyclic_y and boundary in ("zero", "normal"):
        expected[:, :2] = 0
        expected[:, -2:] = 0
        if boundary == "normal":
            expected[:, 2] = 0
    return expected


def _assert_states_equal(actual: State, expected: State) -> None:
    """Check all fields exactly and report one aggregate mismatch per failure."""
    assert type(actual) is type(expected)
    for item in fields(expected):
        first = np.asarray(getattr(actual, item.name))
        second = np.asarray(getattr(expected, item.name))
        assert first.shape == second.shape, f"ERROR {item.name} shape changed"
        assert first.dtype == second.dtype, f"ERROR {item.name} dtype changed"
        assert np.array_equal(first, second), (
            f"ERROR {item.name} preset mismatch: "
            f"max absolute difference={np.max(np.abs(first - second)):.6g}"
        )


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("island", "float64"),
        ("island", "float32"),
        ("dynamics", "float64"),
        ("growth", "float64"),
    ],
)
def test_builtin_initialization_preserves_every_legacy_field(
    name: str, dtype: str
) -> None:
    """Applying custom cleanup to presets would change their historical fields."""
    constants = {"rhoIce": 910.0, "rhoSnow": 310.0}
    settings: dict[str, Any] = {"dtype": dtype, "nEVPsteps": 2}
    scenario: dict[str, Any] = {}
    grid = veris.Grid(2, 2) if name == "growth" else veris.Grid(4, 6)
    if name == "island":
        scenario = {
            "islandGridSpacing": 2000.0,
            "islandWindSpeed": -3.0,
            "islandAirTemperature": 263.0,
            "islandIceThickness": 1.2,
            "islandSnowThickness": 0.07,
            "islandIceArea": 0.65,
            "saltOcn_ref": 33.0,
        }
        expected, expected_conf, expected_phys = island.initialize(
            grid.nx,
            grid.ny,
            settings_overrides=settings,
            physical_overrides=constants,
            scenario_overrides=scenario,
        )
    elif name == "dynamics":
        scenario = {"length": 12000.0, "wind_speed": 7.0, "ice_thickness": 0.4}
        expected, expected_conf, expected_phys = run_dyn.initialize(
            grid.nx,
            grid.ny,
            settings_overrides=settings,
            physical_overrides=constants,
            scenario_overrides=scenario,
        )
    else:
        settings.update(deltatDyn=7200.0, deltatTherm=3600.0)
        expected, expected_conf, expected_phys = run_growth.initialize(
            settings_overrides=settings, physical_overrides=constants
        )
    definition = veris.Experiment.builtin(
        name,
        grid=grid,
        settings=settings,
        constants=constants,
        scenario=scenario,
        steps=3,
    )
    actual, conf, phys = veris.initialize_experiment(definition, mesh=None)
    _assert_states_equal(actual, expected)
    assert conf == expected_conf == definition.configuration
    assert phys == expected_phys == definition.physical_constants
    assert definition.steps == 3
    assert (
        definition.physics
        == {
            "island": "coupled",
            "dynamics": "dynamics",
            "growth": "thermodynamics",
        }[name]
    )


def test_presets_record_prescribed_island_and_recursive_growth_flux_policy() -> None:
    """Preset heat inputs are queued separately from initial coupling outputs."""
    coupled = veris.Experiment.builtin("island", grid=veris.Grid(4, 6))
    assert coupled.resolved_forcing.at(0)["Qnet"] == 100.0
    assert coupled.resolved_forcing.at(0)["Qsw"] == 0.0
    state, _, _ = veris.initialize_experiment(coupled)
    np.testing.assert_array_equal(state.Qnet, 0.0)
    np.testing.assert_array_equal(state.Qsw, 0.0)
    growth = veris.Experiment.builtin("growth", grid=veris.Grid(2, 2))
    assert "Qnet" not in growth.resolved_forcing.at(0)
    assert "Qsw" not in growth.resolved_forcing.at(0)


def test_growth_scenario_initializes_custom_column_without_source_edits() -> None:
    """Scenario coefficients remain separate from numerical and physical values."""
    definition = veris.Experiment.builtin(
        "growth",
        scenario={
            "hIceMean": 0.7,
            "hSnowMean": 0.02,
            "Area": 0.6,
            "oceanTemperatureC": -1.5,
            "ATemp": 260.0,
        },
    )
    state, _, phys = veris.initialize_experiment(definition)
    for name, value in (
        ("hIceMean", 0.7),
        ("hSnowMean", 0.02),
        ("Area", 0.6),
        ("ATemp", 260.0),
        ("theta", phys.celsius2K - 1.5),
        ("SeaIceLoad", phys.rhoIce * 0.7 + phys.rhoSnow * 0.02),
    ):
        np.testing.assert_array_equal(getattr(state, name), value)


def test_explicit_preset_forcing_initializes_fields_and_queues_coupled_heat() -> None:
    """Applying coupled heat early must not replace legacy initial flux outputs."""
    definition = veris.Experiment.builtin(
        "island",
        grid=veris.Grid(4, 6),
        forcing=veris.Forcing(
            constant={
                "theta": 272.0,
                "uWind": 2.0,
                "vWind": 0.0,
                "Qnet": 80.0,
                "Qsw": 2.0,
            }
        ),
    )
    state, _, _ = veris.initialize_experiment(definition)
    for name, value in (
        ("theta", 272.0),
        ("uWind", 2.0),
        ("vWind", 0.0),
        ("wSpeed", 2.0),
    ):
        np.testing.assert_array_equal(getattr(state, name), value)
    np.testing.assert_array_equal(state.Qnet, 0.0)
    np.testing.assert_array_equal(state.Qsw, 0.0)
    assert definition.resolved_forcing.at(0)["Qnet"] == 80.0
    assert definition.resolved_forcing.at(0)["Qsw"] == 2.0


@pytest.mark.parametrize("cyclic_y", [True, False], ids=["periodic", "closed-y"])
def test_custom_geometry_initialization_prepares_consistent_fields_and_halos(
    physical_geometry: OceanGeometry, cyclic_y: bool
) -> None:
    """Custom physical inputs must become runnable, correctly masked C-grid fields."""
    x, y = np.indices((4, 5))
    ice = 1.0 + 0.1 * x + 0.02 * y
    temperature = 259.0 + 0.5 * x + y
    wind_u, wind_v = 3.0 + 0.1 * x, -4.0 + 0.2 * y
    definition = veris.Experiment(
        grid=veris.Grid(4, 5),
        geometry=physical_geometry,
        settings={"enable_cyclic_y": cyclic_y, "geometrySurfaceTemperature": 267.0},
        constants={"rhoIce": 910.0, "rhoSnow": 310.0},
        initial_conditions={
            "hIceMean": ice,
            "hSnowMean": 0.07,
            "Area": 0.65,
            "TSurf": temperature,
            "uIce": 0.01,
            "vIce": -0.02,
        },
        forcing=veris.Forcing(
            constant={
                "theta": 271.25,
                "ocSalt": 33.0,
                "ATemp": 257.0,
                "uWind": wind_u,
                "vWind": wind_v,
                "Qnet": 999.0,
                "Qsw": 4.0,
            },
            series={"Qnet": [80.0, 120.0]},
        ),
        steps=2,
    )
    state, conf, phys = veris.initialize_experiment(definition)
    assert conf == definition.configuration and phys == definition.physical_constants
    assert not conf.use_sharding and (conf.nx, conf.ny) == (4, 5)
    for leaf in (getattr(state, item.name) for item in fields(state)):
        assert leaf.shape == (8, 9) and leaf.dtype == np.dtype(conf.dtype)
        assert np.isfinite(leaf).all(), "ERROR nonfinite custom initial State"
    surface = np.asarray(physical_geometry.maskT)[:, :, -1]
    wet = surface != 0
    expected_ice = np.where(wet, ice, 0)
    expected_snow = np.where(wet, 0.07, 0)
    for name, physical in (
        ("hIceMean", expected_ice),
        ("hSnowMean", expected_snow),
        ("Area", np.where(wet, 0.65, 0)),
    ):
        np.testing.assert_array_equal(
            getattr(state, name), _halo(physical, cyclic_y, "zero")
        )
    # Fractional masks mark wet cells; supplied cell-mean ice is not scaled again.
    assert state.hIceMean[4, 5] == ice[2, 3]
    for source, target, boundary in (
        ("maskT", "iceMask", "zero"),
        ("maskU", "iceMaskU", "zero"),
        ("maskV", "iceMaskV", "normal"),
    ):
        expected_mask = _halo(
            np.asarray(getattr(physical_geometry, source))[:, :, -1], cyclic_y, boundary
        )
        np.testing.assert_array_equal(getattr(state, target), expected_mask)
        np.testing.assert_array_equal(
            getattr(
                state,
                {"iceMask": "maskInC", "iceMaskU": "maskInU", "iceMaskV": "maskInV"}[
                    target
                ],
            ),
            expected_mask,
        )
    for velocity, mask in (("uIce", "maskU"), ("vIce", "maskV")):
        blocked = np.asarray(getattr(physical_geometry, mask))[:, :, -1] == 0
        actual = np.asarray(getattr(state, velocity))[2:-2, 2:-2]
        np.testing.assert_array_equal(actual[blocked], 0)
    expected_load = phys.rhoIce * expected_ice + phys.rhoSnow * expected_snow
    np.testing.assert_array_equal(state.SeaIceLoad, _halo(expected_load, cyclic_y))
    np.testing.assert_allclose(
        state.recip_hIceMean,
        _halo(1 / np.sqrt(expected_ice**2 + phys.hIce_reg), cyclic_y),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(state.TSurf, _halo(temperature, cyclic_y))
    for name, value in (
        ("theta", 271.25),
        ("ocSalt", 33.0),
        ("ATemp", 257.0),
        ("Qnet", 80.0),
        ("Qsw", 4.0),
    ):
        np.testing.assert_array_equal(getattr(state, name), value)
    np.testing.assert_array_equal(state.uWind, _halo(wind_u, cyclic_y))
    np.testing.assert_array_equal(state.vWind, _halo(wind_v, cyclic_y))
    np.testing.assert_allclose(
        state.wSpeed, _halo(np.hypot(wind_u, wind_v), cyclic_y), rtol=1e-14
    )
    if not cyclic_y:
        np.testing.assert_array_equal(state.vIce[:, 2], 0)
        np.testing.assert_array_equal(state.vIce[:, :2], 0)
        np.testing.assert_array_equal(state.vIce[:, -2:], 0)
        np.testing.assert_array_equal(state.uIce[:, :2], 0)
        np.testing.assert_array_equal(state.uIce[:, -2:], 0)
    for name, source in (("R_low", "ht"), ("fCori", "coriolis_t")):
        np.testing.assert_array_equal(
            getattr(state, name),
            _halo(np.asarray(getattr(physical_geometry, source)), cyclic_y),
        )
    for output, source, axis in (
        ("dxG", "dxu", 0),
        ("dyG", "dyu", 1),
        ("dxU", "dxu", 0),
        ("dyU", "dyu", 1),
        ("dxV", "dxt", 0),
        ("dyV", "dyt", 1),
        ("recip_dxC", "dxt", 0),
        ("recip_dyC", "dyt", 1),
    ):
        spacing = np.asarray(getattr(physical_geometry, source))
        metric = np.broadcast_to(spacing[:, None] if axis == 0 else spacing, (4, 5))
        expected = 1 / metric if output.startswith("recip_") else metric
        np.testing.assert_allclose(
            getattr(state, output), _halo(expected, cyclic_y), rtol=1e-14
        )
    area = np.asarray(physical_geometry.area_t)
    expected_corner = np.empty_like(area)
    for i, j in np.ndindex(area.shape):
        south = (j - 1) % 5 if cyclic_y else max(j - 1, 0)
        expected_corner[i, j] = (
            area[i, j] + area[i - 1, j] + area[i, south] + area[i - 1, south]
        ) / 4
    np.testing.assert_allclose(state.rAz[2:-2, 2:-2], expected_corner, rtol=1e-14)
    for output, source in (
        ("recip_rA", "area_t"),
        ("recip_rAu", "area_u"),
        ("recip_rAv", "area_v"),
    ):
        np.testing.assert_allclose(
            getattr(state, output),
            _halo(1 / np.asarray(getattr(physical_geometry, source)), cyclic_y),
            rtol=1e-14,
        )


@pytest.mark.parametrize("explicit_speed", [None, 2.0], ids=["derived", "prescribed"])
def test_initial_wind_speed_uses_first_series_vectors_or_explicit_speed(
    physical_geometry: OceanGeometry, explicit_speed: float | None
) -> None:
    """Series wind overrides constants before deriving its initial magnitude."""
    constant = {"uWind": 7.0, "vWind": -4.0}
    if explicit_speed is not None:
        constant["wSpeed"] = explicit_speed
    definition = veris.Experiment(
        grid=veris.Grid(4, 5),
        geometry=physical_geometry,
        forcing=veris.Forcing(constant=constant, series={"uWind": [3.0, 6.0]}),
    )
    state, _, _ = veris.initialize_experiment(definition)
    np.testing.assert_array_equal(state.uWind, 3.0)
    np.testing.assert_array_equal(state.wSpeed, 5.0 if explicit_speed is None else 2.0)


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("maskT", np.ones((4, 5)), "maskT.*shape"),
        ("maskU", np.ones((4, 5, 0)), "maskU.*nonempty|maskU.*shape"),
        ("maskV", np.ones((4, 4, 2)), "maskV.*shape"),
        ("maskT", np.full((4, 5, 2), 1.1), "maskT|mask.*between"),
        ("maskU", np.full((4, 5, 2), -0.1), "maskU|mask.*between"),
        ("dxt", np.ones((4, 1)), "dxt.*shape"),
        ("dyu", np.zeros(5), "dyu.*positive"),
        ("ht", np.full((5, 4), -100.0), "ht.*shape"),
        ("area_t", np.full((4, 5), -1.0), "area_t.*positive"),
    ],
)
def test_invalid_custom_geometry_is_rejected(
    physical_geometry: OceanGeometry, name: str, value: np.ndarray, message: str
) -> None:
    """A malformed physical grid must not silently become numerical defaults."""
    with pytest.raises(ValueError, match=message):
        definition = veris.Experiment(
            grid=veris.Grid(4, 5), geometry=replace(physical_geometry, **{name: value})
        )
        veris.initialize_experiment(definition)


def test_custom_geometry_dimensions_must_match_global_grid(
    physical_geometry: OceanGeometry,
) -> None:
    """The legacy geometry adapter must not override a conflicting public Grid."""
    with pytest.raises(ValueError, match="geometry|maskT|global.*shape"):
        definition = veris.Experiment(grid=veris.Grid(5, 4), geometry=physical_geometry)
        veris.initialize_experiment(definition)


def test_allocation_defaults_are_not_a_runnable_custom_experiment() -> None:
    """Custom initialization requires geometry instead of unit-metric allocation."""
    with pytest.raises(ValueError, match="geometry|builtin|preset"):
        veris.initialize_experiment(veris.Experiment())


@pytest.mark.parametrize("name", ["unknown", "ocean"])
def test_unknown_builtin_has_informative_error(name: str) -> None:
    with pytest.raises(ValueError, match="builtin|preset|unknown"):
        veris.Experiment.builtin(name)


def test_island_builtin_keeps_its_minimum_physical_grid() -> None:
    with pytest.raises(ValueError, match="four|4.*cell|at least"):
        definition = veris.Experiment.builtin("island", grid=veris.Grid(2, 4))
        veris.initialize_experiment(definition)


@pytest.mark.parametrize(
    "backend, message", [("cpu", "serial|one partition"), ("gpu", "backend")]
)
def test_mesh_request_checks_backend_and_single_partition_semantics(
    backend: str, message: str
) -> None:
    """An explicit mesh must agree with the resource request before allocation."""
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    definition = veris.Experiment.builtin(
        "dynamics",
        grid=veris.Grid(2, 2),
        execution=veris.Execution(backend=backend),
    )
    with pytest.raises(ValueError, match=message):
        veris.initialize_experiment(definition, mesh=mesh)


def test_growth_ocean_temperature_must_remain_positive_kelvin() -> None:
    """Converting scenario Celsius to Kelvin must respect resolved constants."""
    with pytest.raises(ValueError, match="theta|oceanTemperatureC|kelvin"):
        definition = veris.Experiment.builtin(
            "growth", scenario={"oceanTemperatureC": -400.0}
        )
        veris.initialize_experiment(definition)


def test_corner_range_validation_uses_neighbor_sums_instead_of_four_times_maximum(
    physical_geometry: OceanGeometry,
) -> None:
    """A single large cell can still have finite float32 corner intermediates."""
    areas = np.asarray(physical_geometry.area_t).copy()
    areas[1, 2] = 1e38
    definition = veris.Experiment(
        grid=veris.Grid(4, 5),
        geometry=replace(physical_geometry, area_t=areas),
        settings={"dtype": "float32"},
    )
    state, _, _ = veris.initialize_experiment(definition)
    assert np.isfinite(state.rAz).all()
    np.testing.assert_allclose(state.rAz[3, 4], 2.5e37, rtol=1e-6)


def test_invalid_precision_is_rejected_before_device_allocation() -> None:
    """Finite host fields must not become inf, zero kelvin or invalid reciprocals."""
    code = """
from dataclasses import replace
import numpy as np
from jax._src import xla_bridge
from veris import Experiment, Forcing, Grid, OceanGeometry, initialize_experiment

geometry = OceanGeometry(
    maskT=np.ones((2, 2, 1)), maskU=np.ones((2, 2, 1)), maskV=np.ones((2, 2, 1)),
    ht=np.full((2, 2), -100.0), coriolis_t=np.zeros((2, 2)),
    dxt=np.ones(2), dxu=np.ones(2), dyt=np.ones(2), dyu=np.ones(2),
    area_t=np.ones((2, 2)), area_u=np.ones((2, 2)), area_v=np.ones((2, 2)),
)
cases = (
    ('Qnet', {'forcing': Forcing(constant={'Qnet': 1e100})}),
    ('Qnet', {'forcing': Forcing(series={'Qnet': [70.0, 1e100]})}),
    ('theta', {'forcing': Forcing(constant={'theta': 1e-100})}),
    ('hIceMean', {'initial_conditions': {'hIceMean': 1e100}}),
    ('hIceMean', {'initial_conditions': {'hIceMean': 1e30}}),
    ('SeaIceLoad', {'initial_conditions': {'hIceMean': 1e37}}),
    ('TSurf', {'initial_conditions': {'TSurf': 1e-100}}),
    ('dxt', {'geometry': replace(geometry, dxt=np.full(2, 1e-40))}),
    ('area_t', {'geometry': replace(geometry, area_t=np.full((2, 2), 1e-40))}),
    ('wSpeed', {'forcing': Forcing(series={'uWind': [3., 3e38], 'vWind': [4., 3e38]})}),
)
assert not xla_bridge.backends_are_initialized(), 'ERROR import initialized devices'
for field, options in cases:
    kwargs = {'grid': Grid(2, 2), 'geometry': geometry, 'settings': {'dtype': 'float32'}}
    kwargs.update(options)
    try:
        initialize_experiment(Experiment(**kwargs))
    except ValueError as error:
        assert field in str(error), f'ERROR missing precision field context: {error}'
    else:
        raise AssertionError(f'ERROR {field} accepted invalid float32 conversion')
    assert not xla_bridge.backends_are_initialized(), f'ERROR allocated before validating {field}'
try:
    initialize_experiment(Experiment(grid=Grid(2, 2), geometry=geometry))
except ValueError as error:
    assert 'float64' in str(error), f'ERROR missing x64 precision context: {error}'
else:
    raise AssertionError('ERROR accepted disabled x64 precision')
assert not xla_bridge.backends_are_initialized(), 'ERROR allocated before validating x64'
print('precision-preflight-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="false"),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR precision preflight process: {result.stderr[-2000:]}"
    )
    assert result.stdout.strip() == "precision-preflight-ok"
