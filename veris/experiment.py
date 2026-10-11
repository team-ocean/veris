"""Immutable host definition of a standalone sea-ice experiment.

This layer validates grid, input and orchestration choices without allocating a
State or selecting JAX devices. Configuration/PhysicalConstants and their
registries continue to own numerical defaults, precision and scalar validation.
Global physical dimensions are adapted to the allocator's local packed-halo
convention. No equations or numerical tolerances are defined here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from veris._typing import OceanGeometry
from veris.configuration import SETTINGS, Configuration
from veris.forcing import Forcing, _freeze_fields, _immutable_array, _validate_shapes
from veris.io.configuration import OutputSettings
from veris.physical_constants import PhysicalConstants

INITIAL_FIELDS = frozenset(
    {
        "hIceMean",
        "hSnowMean",
        "Area",
        "TSurf",
        "uIce",
        "vIce",
        "sigma1",
        "sigma2",
        "sigma12",
    }
)


def _partitions(value: tuple[int, int]) -> tuple[int, int]:
    """Validate partition counts without enumerating execution devices."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("partitions must be a tuple of two positive integers")
    if any(type(count) is not int or count < 1 for count in value):
        raise ValueError("partitions must contain two positive integers")
    return value


@dataclass(frozen=True)
class Grid:
    """Global physical grid extents in (x, y) order, excluding all halos.

    Staggered center/face/corner fields share the same storage lengths. Existing
    Configuration nx/ny still describe local physical cells per partition.
    """

    nx: int = cast(int, SETTINGS["nx"].default)
    ny: int = cast(int, SETTINGS["ny"].default)

    def __post_init__(self) -> None:
        """Require at least two physical cells along each global direction."""
        for name in ("nx", "ny"):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer of at least two cells")

    def local_shape(self, partitions: tuple[int, int] = (1, 1)) -> tuple[int, int]:
        """Return local physical extents after divisibility/minimum checks."""
        px, py = _partitions(partitions)
        if self.nx % px or self.ny % py:
            raise ValueError("global grid dimensions must be divisible by partitions")
        shape = (self.nx // px, self.ny // py)
        if min(shape) < 2:
            raise ValueError("every partition must contain at least two physical cells")
        return shape

    def storage_shape(self, partitions: tuple[int, int] = (1, 1)) -> tuple[int, int]:
        """Return packed global storage with two halos per partition edge."""
        local_x, local_y = self.local_shape(partitions)
        return (partitions[0] * (local_x + 4), partitions[1] * (local_y + 4))


@dataclass(frozen=True)
class Execution:
    """Execution resource request, separate from static numerical settings.

    backend=None honors the caller's JAX choice; cpu/gpu requests that backend.
    Partitions specify the explicit x/y mesh shape, with (1,1) serial by default.
    Construction never initializes distributed JAX or enumerates devices.
    """

    backend: str | None = None
    partitions: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        """Reject unsupported backends and malformed mesh dimensions."""
        if self.backend not in (None, "cpu", "gpu"):
            raise ValueError("backend must be None, cpu or gpu")
        _partitions(self.partitions)


@dataclass(frozen=True)
class Experiment:
    """One experiment's independent numerical, physical, input and host controls.

    settings and constants are overrides passed to existing registry-backed
    constructors. Grid owns global nx/ny; Execution owns use_sharding. Physical
    constants inherit the numerical dtype. Initial conditions and forcing use
    global physical arrays without halos, in VARIABLES units. Geometry is host
    input to runnable initialization, rather than an array default allocation.

    physics selects coupled, dynamics or thermodynamics. Coupled execution
    requires equal dynamics and thermodynamic timesteps; subcycling is not
    implied. steps specifies the total requested trajectory. output=None disables
    scheduled output. Definition construction allocates no model/device arrays.
    """

    grid: Grid = field(default_factory=Grid)
    settings: Mapping[str, Any] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    physics: str = "coupled"
    geometry: OceanGeometry | None = None
    initial_conditions: Mapping[str, Any] = field(default_factory=dict)
    forcing: Forcing = field(default_factory=Forcing)
    execution: Execution = field(default_factory=Execution)
    steps: int = 0
    output: OutputSettings | None = None
    preset: str | None = None
    scenario: Mapping[str, Any] = field(default_factory=dict)
    configuration: Configuration = field(init=False)
    physical_constants: PhysicalConstants = field(init=False)

    def __post_init__(self) -> None:
        """Validate grouped controls, resolve registry objects and freeze inputs."""
        for name, expected in (
            ("grid", Grid),
            ("execution", Execution),
            ("forcing", Forcing),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if self.physics not in ("coupled", "dynamics", "thermodynamics"):
            raise ValueError("physics must be coupled, dynamics or thermodynamics")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if self.geometry is not None and not isinstance(self.geometry, OceanGeometry):
            raise TypeError("geometry must be OceanGeometry or None")
        if self.geometry is not None:
            owned: dict[str, Any] = {}
            for item in fields(self.geometry):
                array = np.asarray(getattr(self.geometry, item.name))
                if array.dtype.kind not in "bfiu" or not np.isfinite(array).all():
                    raise ValueError(
                        f"geometry {item.name} must contain finite real values"
                    )
                owned[item.name] = _immutable_array(array)
            object.__setattr__(self, "geometry", OceanGeometry(**owned))
        if self.output is not None and not isinstance(self.output, OutputSettings):
            raise TypeError("output must be OutputSettings or None")
        if not isinstance(self.settings, Mapping) or not isinstance(
            self.constants, Mapping
        ):
            raise TypeError("settings and constants must be mappings")
        if {"nx", "ny"} & self.settings.keys():
            raise ValueError("use Grid for global nx/ny, not numerical settings")
        if "use_sharding" in self.settings:
            raise ValueError("use Execution for sharding, not numerical settings")
        if not isinstance(self.scenario, Mapping):
            raise TypeError("scenario must be a mapping")
        if self.preset is not None:
            from veris.presets import PRESET_PHYSICS, preset_controls

            settings, scenario = preset_controls(
                self.preset, self.settings, self.scenario
            )
            if self.physics != PRESET_PHYSICS[self.preset]:
                raise ValueError("physics is incompatible with builtin preset")
            if self.geometry is not None:
                raise ValueError("builtin preset and custom geometry are exclusive")
            object.__setattr__(self, "settings", settings)
            object.__setattr__(self, "scenario", MappingProxyType(scenario))
            if self.preset == "island":
                if min(self.grid.nx, self.grid.ny) < 4:
                    raise ValueError("island requires at least four physical cells")
                object.__setattr__(
                    self,
                    "forcing",
                    Forcing(
                        constant={
                            "Qnet": scenario["islandCooling"],
                            "Qsw": 0.0,
                            **self.forcing.constant,
                        },
                        series=self.forcing.series,
                    ),
                )
        elif self.scenario:
            raise ValueError("scenario controls require a builtin preset")
        else:
            object.__setattr__(self, "scenario", MappingProxyType({}))
        local_x, local_y = self.grid.local_shape(self.execution.partitions)
        conf = Configuration(
            **self.settings,
            nx=local_x,
            ny=local_y,
            use_sharding=self.execution.partitions != (1, 1),
        )
        if "dtype" in self.constants and self.constants["dtype"] != conf.dtype:
            raise ValueError("physical constants dtype must match numerical dtype")
        physical_options: dict[str, Any] = {**self.constants, "dtype": conf.dtype}
        phys = PhysicalConstants(**physical_options)
        if self.physics == "coupled" and conf.deltatDyn != conf.deltatTherm:
            raise ValueError(
                "coupled physics requires equal dynamics/thermodynamic timesteps"
            )
        object.__setattr__(self, "configuration", conf)
        object.__setattr__(self, "physical_constants", phys)
        object.__setattr__(
            self,
            "settings",
            MappingProxyType({name: getattr(conf, name) for name in self.settings}),
        )
        object.__setattr__(
            self,
            "constants",
            MappingProxyType({name: getattr(phys, name) for name in self.constants}),
        )
        initial = _freeze_fields(self.initial_conditions, INITIAL_FIELDS)
        shape = (self.grid.nx, self.grid.ny)
        _validate_shapes(initial, shape)
        object.__setattr__(self, "initial_conditions", initial)
        self.forcing.validate_shape(shape)
        if self.forcing.length is not None and self.steps > self.forcing.length:
            raise ValueError("requested steps exceed forcing series length")
        # Validate the host clock before any resource allocation.
        _ = self.step_duration

    @classmethod
    def builtin(cls, name: str, **options: Any) -> "Experiment":
        """Define island, dynamics or growth using the maintained setup defaults.

        Grid defaults come from the public Grid registry, except the growth
        column uses its scenario's 2 by 2 grid. Explicit options override preset
        settings; scenario contains only experiment-specific coefficients.
        """
        from veris.presets import PRESET_PHYSICS

        if name not in PRESET_PHYSICS:
            raise ValueError(f"unknown builtin preset {name!r}")
        options.setdefault("physics", PRESET_PHYSICS[name])
        if name == "growth" and "grid" not in options:
            from veris.setups.run_growth import GrowthSettings

            controls = GrowthSettings()
            options["grid"] = Grid(controls.nx, controls.ny)
        return cls(preset=name, **options)

    @classmethod
    def from_file(
        cls, path: str | Path, *, overrides: Mapping[str, Any] | None = None
    ) -> "Experiment":
        """Read versioned JSON with registry < file < Python override precedence."""
        from veris.experiment_file import read_experiment

        return read_experiment(path, overrides=overrides)

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible data with resolved scalar defaults."""
        from veris.experiment_file import experiment_to_dict

        return experiment_to_dict(self)

    @property
    def step_duration(self) -> timedelta:
        """Active timestep rounded only within model precision to microseconds.

        Sub-microsecond timesteps and values not representable by the host clock
        fail clearly; insignificant float32 scalar normalization is accepted.
        """
        conf = self.configuration
        seconds = float(
            conf.deltatTherm if self.physics == "thermodynamics" else conf.deltatDyn
        )
        if seconds > timedelta.max.total_seconds():
            raise ValueError("active timestep exceeds the host clock range")
        microseconds = round(seconds * 1_000_000)
        tolerance = np.finfo(conf.dtype).eps * seconds
        if microseconds < 1 or abs(seconds - microseconds / 1_000_000) > tolerance:
            raise ValueError(
                "active timestep must be representable in positive microseconds"
            )
        try:
            return timedelta(microseconds=microseconds)
        except OverflowError as error:
            raise ValueError("active timestep exceeds the host clock range") from error
