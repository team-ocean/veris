"""Runnable host initialization of a validated standalone experiment.

Custom geometry uses global physical cells, with the ocean surface at the last
mask level. The existing ocean adapter supplies the original metric equations;
this layer prepares halos, applies physical input and computes ice load and the
Growth regularized inverse thickness. Presets retain their legacy intermediates.
All validation and resources remain outside the array-only numerical State.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from numpy.typing import NDArray

from veris.grid.geometry import OceanGeometry
from veris.grid.halos import fill_state_overlap
from veris.grid.ocean import _validate_geometry, initialize_from_ocean
from veris.grid.regions import regional_geometry, regional_state
from veris.model.configuration import Configuration
from veris.model.experiment import Experiment
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.model.variables import VARIABLES

_METRICS = frozenset({"dxt", "dxu", "dyt", "dyu", "area_t", "area_u", "area_v"})


def _precision(name: str, value: Any, dtype: str, *, reciprocal: bool = False) -> None:
    """Reject lossy range conversions before JAX can allocate any device arrays."""
    original = np.asarray(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        converted = original.astype(dtype)
        invalid = not np.isfinite(converted).all() or np.any(
            (original != 0) & (converted == 0)
        )
        if reciprocal:
            invalid = invalid or not np.isfinite(1 / converted).all()
    if invalid:
        raise ValueError(
            f"{name} is not representable with valid reciprocals in {dtype}"
        )


def _validate_inputs(experiment: Experiment) -> None:
    """Preflight all series values and geometry before resource selection."""
    dtype = experiment.configuration.dtype
    for inputs in (
        experiment.initial_conditions,
        experiment.forcing.constant,
        experiment.forcing.series,
        experiment.scenario,
    ):
        for name, value in inputs.items():
            _precision(name, value, dtype)
    _validate_derived_inputs(experiment)
    geometry = experiment.geometry
    if geometry is None:
        if experiment.preset is None:
            raise ValueError(
                "runnable initialization requires geometry or a builtin preset"
            )
        return
    _validate_geometry(geometry, (experiment.grid.nx, experiment.grid.ny))
    for item in fields(geometry):
        value = np.asarray(getattr(geometry, item.name))
        if item.name.startswith("mask") and np.any((value < 0) | (value > 1)):
            raise ValueError(f"{item.name} masks must lie between zero and one")
        _precision(item.name, value, dtype, reciprocal=item.name in _METRICS)
    # Preserve the original corner-area arithmetic, including its intermediate
    # sums. Catch finite input values whose sum would overflow that precision.
    with np.errstate(over="ignore", invalid="ignore"):
        area = np.asarray(geometry.area_t, dtype=dtype)
        west_sum = area + np.roll(area, 1, axis=0)
        south_sum = np.roll(west_sum, 1, axis=1)
        if not experiment.configuration.enable_cyclic_y:
            south_sum[:, 0] = west_sum[:, 0]
        if not np.isfinite(west_sum + south_sum).all():
            raise ValueError(f"area_t corner sums are not representable in {dtype}")


def _validate_derived_inputs(experiment: Experiment) -> None:
    """Check load, inverse thickness and wind magnitude in the model precision."""
    conf, phys = experiment.configuration, experiment.physical_constants
    scenario = experiment.scenario
    if experiment.preset == "growth":
        theta = phys.celsius2K + scenario["oceanTemperatureC"]
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError(
                "growth oceanTemperatureC must produce positive theta in kelvin"
            )
    defaults = {
        "hIceMean": scenario.get(
            "islandIceThickness",
            scenario.get(
                "ice_thickness", scenario.get("hIceMean", VARIABLES["hIceMean"].default)
            ),
        ),
        "hSnowMean": scenario.get(
            "islandSnowThickness",
            scenario.get("hSnowMean", VARIABLES["hSnowMean"].default),
        ),
    }
    ice, snow = (
        np.asarray(experiment.initial_conditions.get(name, value), dtype=conf.dtype)
        for name, value in defaults.items()
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        load = ice * phys.rhoIce + snow * phys.rhoSnow
        if not np.isfinite(load).all():
            raise ValueError(f"SeaIceLoad is not representable in {conf.dtype}")
        inverse = 1 / np.sqrt(ice**2 + phys.hIce_reg)
        if not np.isfinite(inverse).all() or np.any(inverse <= 0):
            raise ValueError(
                f"hIceMean regularized inverse is not representable in {conf.dtype}"
            )
    forcing = experiment.forcing
    keys = forcing.constant.keys() | forcing.series.keys()
    if not {"uWind", "vWind"} & keys or "wSpeed" in keys:
        return

    def wind(name: str) -> NDArray[Any]:
        if name in forcing.series:
            value = np.asarray(forcing.series[name], dtype=conf.dtype)
            return value[:, None, None] if value.ndim == 1 else value
        value = np.asarray(forcing.constant.get(name, 0.0), dtype=conf.dtype)
        return value.reshape(1, 1, 1) if value.ndim == 0 else value[None, :, :]

    with np.errstate(over="ignore", invalid="ignore"):
        if not np.isfinite(np.hypot(wind("uWind"), wind("vWind"))).all():
            raise ValueError(
                f"wSpeed vector magnitude is not representable in {conf.dtype}"
            )


def _pad_geometry(geometry: OceanGeometry, conf: Configuration) -> OceanGeometry:
    """Extend physical metrics before the adapter averages adjacent cells."""
    x = (np.arange(conf.nx + 4) - 2) % conf.nx
    y = np.arange(conf.ny + 4) - 2
    y = y % conf.ny if conf.enable_cyclic_y else np.clip(y, 0, conf.ny - 1)
    return regional_geometry(geometry, x, y)


def _stored_field(value: Any, conf: Configuration) -> NDArray[Any]:
    """Convert scalar/global physical data to serial halo-inclusive storage."""
    value = np.asarray(value, dtype=conf.dtype)
    if value.ndim == 0:
        return np.full((conf.nx + 4, conf.ny + 4), value, dtype=conf.dtype)
    x = (np.arange(conf.nx + 4) - 2) % conf.nx
    y = np.arange(conf.ny + 4) - 2
    y = y % conf.ny if conf.enable_cyclic_y else np.clip(y, 0, conf.ny - 1)
    return value[np.ix_(x, y)]


def _apply_inputs(
    state: State,
    initial: Mapping[str, Any],
    forcing: Mapping[str, Any],
    conf: Configuration,
    phys: PhysicalConstants,
    *,
    custom: bool,
    stored: bool = False,
    refresh: bool = True,
) -> State:
    """Apply user fields and refresh derived ice quantities where required."""
    values = {
        name: (
            jnp.full_like(getattr(state, name), value)
            if stored and np.ndim(value) == 0
            else jnp.asarray(value, dtype=conf.dtype)
            if stored
            else jnp.asarray(_stored_field(value, conf))
        )
        for name, value in {**forcing, **initial}.items()
    }
    state = replace(state, **values)
    if {"uWind", "vWind"} & forcing.keys() and "wSpeed" not in forcing:
        state = replace(state, wSpeed=jnp.hypot(state.uWind, state.vWind))
    amounts = {"hIceMean", "hSnowMean", "Area"}
    if custom or amounts & initial.keys():
        state = replace(
            state,
            **{
                name: jnp.where(state.iceMask != 0, getattr(state, name), 0)
                for name in amounts
                if custom or name in initial
            },
        )
        state = replace(
            state,
            SeaIceLoad=state.hIceMean * phys.rhoIce + state.hSnowMean * phys.rhoSnow,
            recip_hIceMean=1 / jnp.sqrt(state.hIceMean**2 + phys.hIce_reg),
        )
    if custom:
        state = replace(
            state,
            uIce=jnp.where(state.iceMaskU != 0, state.uIce, 0),
            vIce=jnp.where(state.iceMaskV != 0, state.vIce, 0),
        )
    return fill_state_overlap(state, conf) if refresh and (custom or values) else state


def _initialize_serial(experiment: Experiment) -> State:
    """Reuse exact setup factories or the established ocean metric adapter."""
    conf, phys = experiment.configuration, experiment.physical_constants
    numerical = dict(experiment.settings)
    numerical["use_sharding"] = False
    physical = dict(experiment.constants)
    forcing = dict(experiment.forcing.at(0))
    if experiment.preset == "island":
        from veris.setups.island import initialize

        scenario = {
            k: v for k, v in experiment.scenario.items() if k != "islandCooling"
        }
        state, _, _ = initialize(
            experiment.grid.nx,
            experiment.grid.ny,
            settings_overrides=numerical,
            physical_overrides=physical,
            scenario_overrides=scenario,
        )
        # Coupling outputs start at their established values. Prescribed heat is
        # an input to advance(), not a replacement for startup coupling outputs.
        forcing.pop("Qnet", None)
        forcing.pop("Qsw", None)
    elif experiment.preset == "dynamics":
        from veris.setups.run_dyn import initialize

        state, _, _ = initialize(
            experiment.grid.nx,
            experiment.grid.ny,
            settings_overrides=numerical,
            physical_overrides=physical,
            scenario_overrides=experiment.scenario,
        )
    elif experiment.preset == "growth":
        from veris.setups.run_growth import initialize

        state, _, _ = initialize(
            settings_overrides={**numerical, "nx": conf.nx, "ny": conf.ny},
            physical_overrides=physical,
            scenario_overrides=experiment.scenario,
        )
    else:
        assert experiment.geometry is not None
        state, _, _ = initialize_from_ocean(
            _pad_geometry(experiment.geometry, conf),
            settings_overrides=numerical,
            physical_overrides=physical,
        )
    return _apply_inputs(
        state,
        experiment.initial_conditions,
        forcing,
        conf,
        phys,
        custom=experiment.preset is None,
    )


def _initialize_sharded(experiment: Experiment, mesh: Mesh) -> State:
    """Build only addressable regions, then establish true global boundary halos."""
    from veris.execution.initialization import initialize as allocate

    conf, phys = experiment.configuration, experiment.physical_constants
    # A regional factory must never close its local north/south edges. Global
    # wall policies are applied to the complete packed layout after callbacks.
    local = replace(conf, use_sharding=False, enable_cyclic_y=True)
    numerical = {
        **experiment.settings,
        "nx": conf.nx,
        "ny": conf.ny,
        "use_sharding": False,
        "enable_cyclic_y": True,
    }
    global_shape = experiment.grid.nx, experiment.grid.ny
    forcing = dict(experiment.forcing.at(0))
    if experiment.preset == "island":
        forcing.pop("Qnet", None)
        forcing.pop("Qsw", None)

    def build(x: Any, y: Any) -> dict[str, Any]:
        if experiment.preset == "dynamics":
            from veris.setups.run_dyn import DynamicsSettings, _fields

            state = State(
                **{
                    name: jnp.asarray(value)
                    for name, value in _fields(
                        x,
                        y,
                        *global_shape,
                        DynamicsSettings(**dict(experiment.scenario)),
                        conf.dtype,
                    ).items()
                }
            )
        elif experiment.preset == "island":
            from veris.setups.island import IslandSettings, _fields

            state, _, _ = allocate(
                settings_overrides=numerical,
                physical_overrides=experiment.constants,
            )
            scenario = IslandSettings(
                dtype=conf.dtype,
                **{
                    k: v for k, v in experiment.scenario.items() if k != "islandCooling"
                },
            )
            state = replace(
                state,
                **_fields(state, phys, scenario, *global_shape, x, y, regional=True),
            )
        elif experiment.preset == "growth":
            from veris.setups.run_growth import initialize

            state, _, _ = initialize(
                settings_overrides=numerical,
                physical_overrides=experiment.constants,
                scenario_overrides=experiment.scenario,
            )
        else:
            assert experiment.geometry is not None
            state, _, _ = initialize_from_ocean(
                regional_geometry(experiment.geometry, x, y),
                settings_overrides=numerical,
                physical_overrides=experiment.constants,
            )

        def region(values: Mapping[str, Any]) -> dict[str, Any]:
            return {
                name: value if np.ndim(value) == 0 else np.asarray(value)[np.ix_(x, y)]
                for name, value in values.items()
            }

        state = _apply_inputs(
            state,
            region(experiment.initial_conditions),
            region(forcing),
            local,
            phys,
            custom=experiment.preset is None,
            stored=True,
            refresh=False,
        )
        return {item.name: getattr(state, item.name) for item in fields(state)}

    state = regional_state(mesh, conf, global_shape, build)
    # Custom metric corner stencils and all user overrides need exchanged
    # halos. Uniform periodic presets already construct their exact halos.
    if (
        not conf.enable_cyclic_y
        or experiment.preset is None
        or experiment.initial_conditions
        or forcing
    ):
        state = fill_state_overlap(state, conf)
    return state


def initialize_experiment(
    experiment: Experiment, *, mesh: Mesh | None = None
) -> tuple[State, Configuration, PhysicalConstants]:
    """Validate and initialize one experiment using global physical dimensions.

    This host operation is outside differentiated computation. Enable JAX x64
    explicitly before allocation when requesting float64. Distributed setup and
    meshes must be prepared explicitly before device enumeration. All presets
    and custom geometry support explicitly supplied meshes using addressable
    regional callbacks, followed by global field-aware halo exchange.
    """
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    _validate_inputs(experiment)
    conf, phys = experiment.configuration, experiment.physical_constants
    if jax.dtypes.canonicalize_dtype(conf.dtype) != jnp.dtype(conf.dtype):
        raise ValueError(
            f"dtype {conf.dtype} requires jax_enable_x64 before initialization"
        )
    if mesh is not None:
        if not isinstance(mesh, Mesh) or set(mesh.axis_names) != {"x", "y"}:
            raise ValueError("mesh must be a JAX Mesh with x and y axes")
        if (mesh.shape["x"], mesh.shape["y"]) != experiment.execution.partitions:
            raise ValueError("mesh shape must match Execution partitions")
        backend = experiment.execution.backend
        platforms = {"cpu"} if backend == "cpu" else {"gpu", "cuda", "rocm"}
        if backend is not None and any(
            device.platform not in platforms for device in mesh.devices.flat
        ):
            raise ValueError("mesh devices do not match Execution backend")
        if not conf.use_sharding:
            raise ValueError(
                "serial execution with one partition does not need an explicit mesh"
            )
    if conf.use_sharding or mesh is not None:
        if mesh is None:
            raise ValueError("sharded initialization requires an explicit mesh")
        with jax.set_mesh(mesh):
            state = _initialize_sharded(experiment, mesh)
    else:
        backend = experiment.execution.backend
        context = (
            nullcontext()
            if backend is None
            else jax.default_device(jax.devices(backend)[0])
        )
        with jax.set_mesh(None), context:
            state = _initialize_serial(experiment)
    return state, conf, phys


__all__ = ["initialize_experiment"]
