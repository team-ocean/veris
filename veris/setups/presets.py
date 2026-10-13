"""Registered scientific setup data and regional field factories.

Scenario registries own scientific literals; profiles own intentional launch
differences. Registration and lookup select no devices and allocate no arrays.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, cast

from veris.model.configuration import SETTINGS
from veris.model.experiment_resolution import freeze_controls
from veris.model.forcing import FORCING_FIELDS, _freeze_fields
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import Parameter

type Controls = Mapping[str, Any]
type ScenarioFactory = Callable[[Controls, str], Controls]
type NumericalDefaults = Controls | Callable[[Controls], Controls]
type InitialDefaults = Callable[[Controls, PhysicalConstants], Controls]
type FieldFactory = Callable[..., Controls]


class _UnknownScenarioError(ValueError):
    """Identify schema errors for legacy Python-argument error translation."""


@dataclass(frozen=True)
class SetupPreset:
    """Host schema, declarative startup policies and a regional data factory.

    field_factory accepts baseline State, Configuration, PhysicalConstants,
    scenario, global shape, x/y indices and keyword regional. It supplies field
    replacements without allocation, global halos or execution resources.
    """

    name: str
    physics: str
    scenario_registry: Mapping[str, Parameter]
    scenario_factory: ScenarioFactory
    field_factory: FieldFactory
    numerical_defaults: NumericalDefaults = field(default_factory=dict)
    grid: tuple[int, int] | None = None
    profiles: Mapping[str, Controls] = field(default_factory=dict)
    minimum_shape: tuple[int, int] = (2, 2)
    forcing_defaults: Controls | Callable[[Controls], Controls] = field(
        default_factory=dict
    )
    initial_defaults: InitialDefaults | None = None
    startup_excluded_forcing: frozenset[str] = frozenset()
    preserve_startup: bool = False

    def __post_init__(self) -> None:
        """Validate provider identity and defensively own declared defaults."""
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name.strip() != self.name
        ):
            raise ValueError("preset name must be a nonempty identifier")
        if self.physics not in ("coupled", "dynamics", "thermodynamics"):
            raise ValueError(
                "preset physics must be coupled, dynamics or thermodynamics"
            )
        if not callable(self.scenario_factory) or not callable(self.field_factory):
            raise TypeError("preset scenario and field factories must be callable")
        for name in ("scenario_registry", "profiles"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"preset {name} must be a mapping")
            object.__setattr__(self, name, freeze_controls(getattr(self, name)))
        allowed_profiles = {
            "grid",
            "steps",
            "execution",
            "settings",
            "constants",
            "scenario",
            "initial_conditions",
            "forcing",
            "output",
        }
        for name, controls in self.profiles.items():
            if not isinstance(controls, Mapping):
                raise TypeError(f"profile {name!r} must be a mapping")
            unknown = controls.keys() - allowed_profiles
            if unknown:
                raise ValueError(
                    f"profile {name!r} has unknown controls: {', '.join(sorted(unknown))}"
                )
        for name in ("numerical_defaults", "forcing_defaults"):
            value = getattr(self, name)
            if not callable(value):
                if not isinstance(value, Mapping):
                    raise TypeError(f"preset {name} must be a mapping or callable")
                object.__setattr__(self, name, freeze_controls(value))
        for name in ("grid", "minimum_shape"):
            shape = getattr(self, name)
            if shape is not None and (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(type(size) is not int or size < 2 for size in shape)
            ):
                raise ValueError(
                    f"preset {name} requires two integer extents of at least two"
                )

    def profile_controls(self, name: str | None) -> Controls:
        """Return launch controls, rejecting unknown profile names."""
        if name is None:
            return {}
        if name not in self.profiles:
            raise ValueError(f"unknown profile {name!r} for preset {self.name!r}")
        return self.profiles[name]

    def resolve_scenario(self, values: Controls, dtype: str) -> Controls:
        """Resolve the schema and validate common prescribed physical quantities."""
        if {"nx", "ny", "steps", "dtype"} & values.keys():
            raise ValueError(
                "use Grid, Experiment steps and settings dtype for controls"
            )
        unknown = values.keys() - self.scenario_registry.keys()
        if unknown:
            raise _UnknownScenarioError(
                f"unknown scenario controls: {', '.join(sorted(unknown))}"
            )
        result = self.scenario_factory(values, dtype)
        allowed = FORCING_FIELDS | {"hIceMean", "hSnowMean", "Area", "TSurf"}
        _freeze_fields(
            {name: value for name, value in result.items() if name in allowed}, allowed
        )
        return freeze_controls(result)

    def grid_defaults(self, profile: str | None = None) -> Controls:
        """Resolve shared, scientific and profile dimensions before a partial grid."""
        result = {name: SETTINGS[name].default for name in ("nx", "ny")}
        if self.grid is not None:
            result.update(nx=self.grid[0], ny=self.grid[1])
        shape = self.profile_controls(profile).get("grid")
        if isinstance(shape, Mapping):
            result.update(shape)
        elif shape is not None:
            result.update(nx=shape[0], ny=shape[1])
        return result


_PRESETS: dict[str, SetupPreset] = {}
_LOADED = False


def _scenario(
    factory: Callable[..., Any], values: Controls, dtype: str | None
) -> Controls:
    """Adapt validated setup dataclasses without inserting run controls."""
    control = factory(**({} if dtype is None else {"dtype": dtype}), **values)
    return {
        item.name: getattr(control, item.name)
        for item in fields(control)
        if item.name not in {"dtype", "nx", "ny", "steps"}
    }


def _load_builtins() -> None:
    """Register maintained providers once with deferred host imports."""
    global _LOADED
    if _LOADED:
        return
    from veris.setups import island, run_dyn, run_growth

    providers = (
        SetupPreset(
            "island",
            "coupled",
            island.ISLAND_SETTINGS,
            lambda values, dtype: _scenario(island.IslandSettings, values, dtype),
            island.build_fields,
            numerical_defaults=lambda values: island.numerical_defaults(
                island.IslandSettings(**values)
            ),
            profiles=island.ISLAND_PROFILES,
            minimum_shape=(4, 4),
            forcing_defaults=lambda values: {
                "Qnet": values["islandCooling"],
                "Qsw": 0.0,
            },
            initial_defaults=lambda values, constants: {
                "hIceMean": values["islandIceThickness"],
                "hSnowMean": values["islandSnowThickness"],
                "Area": values["islandIceArea"],
            },
            startup_excluded_forcing=frozenset({"Qnet", "Qsw"}),
            preserve_startup=True,
        ),
        SetupPreset(
            "dynamics",
            "dynamics",
            run_dyn.DYNAMICS_SETTINGS,
            lambda values, dtype: _scenario(run_dyn.DynamicsSettings, values, None),
            run_dyn.build_fields,
            numerical_defaults=run_dyn.DYNAMICS_NUMERICAL_DEFAULTS,
            preserve_startup=True,
            profiles=run_dyn.DYNAMICS_PROFILES,
            initial_defaults=lambda values, constants: {
                "hIceMean": values["ice_thickness"],
                "hSnowMean": 0.0,
            },
        ),
        SetupPreset(
            "growth",
            "thermodynamics",
            run_growth.GROWTH_SETTINGS,
            lambda values, dtype: _scenario(run_growth.GrowthSettings, values, dtype),
            run_growth.build_fields,
            numerical_defaults=run_growth.GROWTH_NUMERICAL_DEFAULTS,
            preserve_startup=True,
            grid=(
                cast(int, run_growth.GROWTH_SETTINGS["nx"].default),
                cast(int, run_growth.GROWTH_SETTINGS["ny"].default),
            ),
            profiles=run_growth.GROWTH_PROFILES,
            initial_defaults=lambda values, constants: {
                "hIceMean": values["hIceMean"],
                "hSnowMean": values["hSnowMean"],
                "Area": values["Area"],
                "theta": constants.celsius2K + values["oceanTemperatureC"],
            },
        ),
    )
    _PRESETS.update({preset.name: preset for preset in providers})
    _LOADED = True


def register_preset(preset: SetupPreset) -> None:
    """Register an explicit provider without replacing another default owner."""
    _load_builtins()
    if not isinstance(preset, SetupPreset):
        raise TypeError("preset must be a SetupPreset")
    if preset.name in _PRESETS:
        raise ValueError(f"preset {preset.name!r} is already registered")
    _PRESETS[preset.name] = preset


def get_preset(name: str) -> SetupPreset:
    """Look up a scientific provider without selecting execution devices."""
    _load_builtins()
    if not isinstance(name, str) or name not in _PRESETS:
        raise ValueError(f"unknown builtin preset {name!r}")
    return _PRESETS[name]


def _initialize_adapter(
    name: str,
    *,
    nx: int | None = None,
    ny: int | None = None,
    dtype: str | None = None,
    steps: Any = 0,
    mesh: Any = None,
    settings: Controls | None = None,
    constants: Controls | None = None,
    scenario: Controls | None = None,
) -> Any:
    """Translate legacy initialization controls into the common Experiment API."""
    from jax.sharding import Mesh

    from veris.execution.experiment_initialization import initialize_experiment
    from veris.model.experiment import Execution, Grid, _InitializationExperiment

    options = dict(settings or {})
    if options.pop("use_sharding", False) and mesh is None:
        raise ValueError("sharded initialization requires an explicit mesh")
    if mesh is not None and (
        not isinstance(mesh, Mesh) or set(mesh.axis_names) != {"x", "y"}
    ):
        raise ValueError("mesh must be a JAX Mesh with x and y axes")
    shape = dict(get_preset(name).grid_defaults())
    for key, explicit in (("nx", nx), ("ny", ny)):
        shape[key] = options.pop(key, shape[key])
        if explicit is not None:
            shape[key] = explicit
    if dtype is not None:
        options["dtype"] = dtype
    partitions = (1, 1) if mesh is None else (mesh.shape["x"], mesh.shape["y"])
    try:
        definition = _InitializationExperiment.builtin(
            name,
            grid=Grid(**shape),
            settings=options,
            constants=constants or {},
            scenario=scenario or {},
            execution=Execution(partitions=partitions),
            _mesh_mode=mesh is not None,
            steps=steps,
        )
    except _UnknownScenarioError as error:
        raise TypeError(str(error)) from error
    return initialize_experiment(definition, mesh=mesh)


def preset_controls(
    name: str, settings: Controls, scenario: Controls, profile: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply scientific/profile defaults before authored numerical controls."""
    from veris.model.experiment_resolution import merge_overrides

    preset = get_preset(name)
    launch = preset.profile_controls(profile)
    combined = merge_overrides(launch.get("settings", {}), settings)
    values = merge_overrides(launch.get("scenario", {}), scenario)
    resolved = preset.resolve_scenario(
        values, combined.get("dtype", SETTINGS["dtype"].default)
    )
    defaults = preset.numerical_defaults
    numerical = defaults if isinstance(defaults, Mapping) else defaults(resolved)
    return {**numerical, **combined}, dict(resolved)
