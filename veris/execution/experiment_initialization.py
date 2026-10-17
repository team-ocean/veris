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

from veris.grid.halos import fill_state_overlap
from veris.grid.ocean import _validate_geometry, initialize_from_ocean
from veris.grid.regions import regional_geometry, regional_state
from veris.model.configuration import Configuration
from veris.model.experiment import Experiment
from veris.model.forcing import FORCING_FIELDS, _freeze_fields, _precision
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.model.variables import VARIABLES
from veris.setups.presets import get_preset

_METRICS = frozenset({"dxt", "dxu", "dyt", "dyu", "area_t", "area_u", "area_v"})


def _validate_inputs(experiment: Experiment) -> None:
    """Preflight all series values and geometry before resource selection."""
    dtype = experiment.configuration.dtype
    for inputs in (
        experiment.resolved_initial_conditions,
        experiment.resolved_forcing.constant,
        experiment.resolved_forcing.series,
        experiment.resolved_scenario,
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
    _validate_geometry(
        geometry, (experiment.grid.nx, experiment.grid.ny), physical=True
    )
    for item in fields(geometry):
        if getattr(geometry, item.name) is None:
            continue
        value = np.asarray(getattr(geometry, item.name))
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
    defaults = {name: VARIABLES[name].default for name in ("hIceMean", "hSnowMean")}
    if experiment.preset is not None:
        provider = get_preset(experiment.preset)
        if provider.initial_defaults is not None:
            supplied = provider.initial_defaults(experiment.resolved_scenario, phys)
            allowed = FORCING_FIELDS | {"hIceMean", "hSnowMean", "Area", "TSurf"}
            _freeze_fields(supplied, allowed)
            for name, value in supplied.items():
                _precision(name, value, conf.dtype)
            defaults.update(supplied)
    defaults = {name: defaults[name] for name in ("hIceMean", "hSnowMean")}
    ice, snow = (
        np.asarray(
            experiment.resolved_initial_conditions.get(name, value), dtype=conf.dtype
        )
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
    forcing = experiment.resolved_forcing
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


def _startup_inputs(experiment: Experiment) -> tuple[dict[str, Any], bool]:
    """Select startup forcing and the provider's derived-field policy."""
    forcing = dict(experiment.resolved_forcing.at(0))
    if experiment.preset is None:
        return forcing, True
    provider = get_preset(experiment.preset)
    for name in provider.startup_excluded_forcing:
        forcing.pop(name, None)
    return forcing, not provider.preserve_startup


def _build_region(
    experiment: Experiment,
    conf: Configuration,
    x: Any,
    y: Any,
    *,
    regional: bool,
) -> State:
    """Allocate a local baseline and apply one registered data factory."""
    from veris.execution.initialization import initialize as allocate

    numerical = {
        **experiment.resolved_settings,
        "nx": conf.nx,
        "ny": conf.ny,
        "use_sharding": False,
        "enable_cyclic_y": conf.enable_cyclic_y,
    }
    if experiment.geometry is not None:
        geometry = regional_geometry(experiment.geometry, x, y)
        if regional and experiment.geometry.latitude is not None:
            # Local periodic halos are communication storage, not periodic
            # latitude. The globally validated spherical metrics are sliced
            # below; the allocation adapter must not close an internal wall.
            geometry = replace(
                geometry, latitude=None, latitude_edges=None, radius=None
            )
        state, _, _ = initialize_from_ocean(
            geometry,
            settings_overrides=numerical,
            physical_overrides=experiment.resolved_constants,
        )
        if experiment.geometry.latitude is not None:
            from veris.grid.spherical import regional_spherical_metrics

            state = replace(
                state,
                **regional_spherical_metrics(
                    experiment.geometry, x, y, dtype=conf.dtype
                ),
            )
        return state
    assert experiment.preset is not None
    provider = get_preset(experiment.preset)
    state, _, phys = allocate(
        settings_overrides=numerical,
        physical_overrides=experiment.resolved_constants,
    )
    replacements = provider.field_factory(
        state,
        conf,
        phys,
        experiment.resolved_scenario,
        (experiment.grid.nx, experiment.grid.ny),
        x,
        y,
        regional=regional,
    )
    if not isinstance(replacements, Mapping):
        raise TypeError("provider field_factory must return a mapping of State fields")
    unknown = replacements.keys() - VARIABLES.keys()
    if unknown:
        raise ValueError(f"unknown provider State fields: {', '.join(sorted(unknown))}")
    if not provider.preserve_startup:
        allowed = FORCING_FIELDS | {"hIceMean", "hSnowMean", "Area", "TSurf"}
        _freeze_fields({k: v for k, v in replacements.items() if k in allowed}, allowed)
    values = {}
    for name, value in replacements.items():
        _precision(name, value, conf.dtype)
        array = jnp.asarray(value, dtype=conf.dtype)
        expected = getattr(state, name).shape
        if array.ndim == 0:
            array = jnp.full(expected, array, dtype=conf.dtype)
        if array.shape != expected:
            raise ValueError(
                f"provider {name} has shape {array.shape}; expected halo-inclusive {expected}"
            )
        values[name] = array
    state = replace(state, **values)
    return fill_state_overlap(state, conf) if not conf.enable_cyclic_y else state


def _initialize_serial(experiment: Experiment) -> State:
    """Build the serial region through the same provider contract as shards."""
    conf, phys = experiment.configuration, experiment.physical_constants
    x = (np.arange(conf.nx + 4) - 2) % conf.nx
    y = np.arange(conf.ny + 4) - 2
    y = (
        np.clip(y, 0, conf.ny - 1)
        if experiment.geometry is not None and not conf.enable_cyclic_y
        else y % conf.ny
    )
    state = _build_region(experiment, conf, x, y, regional=False)
    forcing, custom = _startup_inputs(experiment)
    return _apply_inputs(
        state,
        experiment.resolved_initial_conditions,
        forcing,
        conf,
        phys,
        custom=custom,
    )


def _initialize_sharded(experiment: Experiment, mesh: Mesh) -> State:
    """Build only addressable regions, then establish true global boundary halos."""
    conf, phys = experiment.configuration, experiment.physical_constants
    # A regional factory never closes its local edges. Global wall policies are
    # applied to the complete packed layout after addressable callbacks finish.
    local = replace(conf, use_sharding=False, enable_cyclic_y=True)
    global_shape = experiment.grid.nx, experiment.grid.ny
    forcing, custom = _startup_inputs(experiment)

    def build(x: Any, y: Any) -> dict[str, Any]:
        state = _build_region(experiment, local, x, y, regional=True)

        def region(values: Mapping[str, Any]) -> dict[str, Any]:
            return {
                name: value if np.ndim(value) == 0 else np.asarray(value)[np.ix_(x, y)]
                for name, value in values.items()
            }

        state = _apply_inputs(
            state,
            region(experiment.resolved_initial_conditions),
            region(forcing),
            local,
            phys,
            custom=custom,
            stored=True,
            refresh=False,
        )
        return {item.name: getattr(state, item.name) for item in fields(state)}

    state = regional_state(mesh, conf, global_shape, build)
    if (
        not conf.enable_cyclic_y
        or custom
        or experiment.resolved_initial_conditions
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
