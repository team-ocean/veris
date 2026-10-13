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
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from veris.grid.geometry import OceanGeometry
from veris.io.configuration import OutputSettings
from veris.model.configuration import SETTINGS, Configuration
from veris.model.experiment_resolution import freeze_controls
from veris.model.forcing import (
    Forcing,
    _freeze_fields,
    _immutable_array,
    _validate_shapes,
)
from veris.model.physical_constants import PhysicalConstants


class _DefaultInput(Enum):
    """Mark omitted constructor controls without confusing zero/None with omission."""

    SHARED = "shared or preset default"


_DEFAULT_INPUT: Any = _DefaultInput.SHARED
_DEFAULTED_CONTROLS = ("grid", "physics", "steps", "execution", "output")

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
class DistributedExecution:
    """Explicit distributed bootstrap controls, validated without JAX devices.

    Supply coordinator/count/rank together, or a supported cluster detector.
    Local device IDs are immutable nonnegative indices. Execution resources
    adapt environment defaults only when no explicit bootstrap is supplied.
    """

    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    cluster_detection_method: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete rendezvous controls and malformed ranks/device IDs."""
        explicit = (self.coordinator_address, self.num_processes, self.process_id)
        if any(value is not None for value in explicit):
            if any(value is None for value in explicit):
                raise ValueError(
                    "coordinator/count/process controls must be set together"
                )
            address = self.coordinator_address
            if not isinstance(address, str):
                raise ValueError(
                    "coordinator_address must contain a host and valid port"
                )
            host, separator, port = address.rpartition(":")
            if (
                not isinstance(address, str)
                or not host
                or not separator
                or any(c.isspace() for c in address)
                or not port.isdecimal()
                or not 1 <= int(port) <= 65535
            ):
                raise ValueError(
                    "coordinator_address must contain a host and valid port"
                )
            if type(self.num_processes) is not int or self.num_processes < 1:
                raise ValueError("num_processes must be a positive integer")
            if (
                type(self.process_id) is not int
                or not 0 <= self.process_id < self.num_processes
            ):
                raise ValueError(
                    "process_id must be an integer within the process count"
                )
            if self.cluster_detection_method is not None:
                raise ValueError(
                    "explicit coordinator controls conflict with cluster detection"
                )
        elif self.cluster_detection_method not in {"slurm", "ompi", "mpi4py", "k8s"}:
            raise ValueError(
                "distributed execution requires coordinator controls or a supported cluster detector"
            )
        ids = self.local_device_ids
        if ids is not None and (
            not isinstance(ids, tuple)
            or not ids
            or any(type(index) is not int or index < 0 for index in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(
                "local device IDs must be a nonempty tuple of distinct nonnegative integers"
            )


@dataclass(frozen=True)
class Execution:
    """Execution resource request, separate from static numerical settings.

    backend=None honors the caller's JAX choice; cpu/gpu requests that backend.
    Partitions specify the explicit x/y mesh shape, with (1,1) serial by default.
    Construction never initializes distributed JAX or enumerates devices.
    """

    backend: str | None = None
    partitions: tuple[int, int] = (1, 1)
    distributed: DistributedExecution | None = None
    all_devices: bool = False

    def __post_init__(self) -> None:
        """Reject unsupported backends and malformed mesh dimensions."""
        if self.backend not in (None, "cpu", "gpu"):
            raise ValueError("backend must be None, cpu or gpu")
        _partitions(self.partitions)
        if type(self.all_devices) is not bool:
            raise TypeError("all_devices must be a bool")
        if self.all_devices and self.partitions != (1, 1):
            raise ValueError("all_devices conflicts with explicit partitions")
        if self.distributed is not None:
            if not isinstance(self.distributed, DistributedExecution):
                raise TypeError("distributed must be DistributedExecution or None")
            if self.backend is None:
                raise ValueError("distributed execution requires an explicit backend")


@dataclass(frozen=True)
class Experiment:
    """One experiment's independent numerical, physical, input and host controls.

    settings, constants, scenario and forcing retain authored inputs. Inherited
    defaults and dtype-normalized values live in configuration,
    physical_constants, resolved_settings, resolved_scenario and resolved_forcing.
    with_overrides merges authored values before resolving them again, preserving
    shared defaults < preset < configuration < explicit overrides precedence.
    Grid owns global nx/ny; Execution owns use_sharding. Physical
    constants inherit the numerical dtype. Initial conditions and forcing use
    global physical arrays without halos, in VARIABLES units. Geometry is host
    input to runnable initialization, rather than an array default allocation.

    physics selects coupled, dynamics or thermodynamics. Coupled execution
    requires equal dynamics and thermodynamic timesteps; subcycling is not
    implied. steps specifies the total requested trajectory. output=None disables
    scheduled output. Definition construction allocates no model/device arrays.
    """

    grid: Grid = _DEFAULT_INPUT
    settings: Mapping[str, Any] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)
    physics: str = _DEFAULT_INPUT
    geometry: OceanGeometry | None = None
    initial_conditions: Mapping[str, Any] = field(default_factory=dict)
    forcing: Forcing = field(default_factory=Forcing)
    execution: Execution = _DEFAULT_INPUT
    steps: int = _DEFAULT_INPUT
    output: OutputSettings | None = _DEFAULT_INPUT
    preset: str | None = None
    profile: str | None = None
    scenario: Mapping[str, Any] = field(default_factory=dict)
    configuration: Configuration = field(init=False)
    physical_constants: PhysicalConstants = field(init=False)
    resolved_settings: Mapping[str, Any] = field(init=False, repr=False)
    resolved_scenario: Mapping[str, Any] = field(init=False, repr=False)
    resolved_forcing: Forcing = field(init=False, repr=False)
    resolved_constants: Mapping[str, Any] = field(init=False, repr=False)
    resolved_initial_conditions: Mapping[str, Any] = field(init=False, repr=False)
    _authored_controls: Mapping[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate grouped controls, resolve registry objects and freeze inputs."""
        from veris.model.experiment_file import _execution, _output
        from veris.setups.presets import get_preset

        authored = {
            name: getattr(self, name)
            for name in _DEFAULTED_CONTROLS
            if getattr(self, name) is not _DEFAULT_INPUT
        }
        provider = None if self.preset is None else get_preset(self.preset)
        if self.profile is not None and provider is None:
            raise ValueError("profile requires a preset")
        launch = {} if provider is None else provider.profile_controls(self.profile)
        defaults = {
            "grid": Grid()
            if provider is None
            else Grid(**provider.grid_defaults(self.profile)),
            "physics": "coupled" if provider is None else provider.physics,
            "steps": launch.get("steps", 0),
            "execution": _execution(launch.get("execution", {})),
            "output": _output(launch.get("output")),
        }
        for name, value in defaults.items():
            if getattr(self, name) is _DEFAULT_INPUT:
                object.__setattr__(self, name, value)
        object.__setattr__(self, "_authored_controls", MappingProxyType(authored))
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
        for name in ("settings", "constants", "scenario"):
            object.__setattr__(self, name, freeze_controls(getattr(self, name)))
        settings = dict(self.settings)
        scenario: dict[str, Any] = {}
        forcing = self.forcing
        if self.preset is not None:
            from veris.setups.presets import preset_controls

            settings, scenario = preset_controls(
                self.preset, self.settings, self.scenario, self.profile
            )
            assert provider is not None
            if self.physics != provider.physics:
                raise ValueError("physics is incompatible with builtin preset")
            if self.geometry is not None:
                raise ValueError("builtin preset and custom geometry are exclusive")
            if any(
                size < minimum
                for size, minimum in zip(
                    (self.grid.nx, self.grid.ny), provider.minimum_shape, strict=True
                )
            ):
                words = {2: "two", 4: "four"}
                minimum = " by ".join(
                    str(words.get(size, size)) for size in provider.minimum_shape
                )
                raise ValueError(
                    f"{provider.name} requires at least {minimum} physical cells"
                )
            prescribed = provider.forcing_defaults
            prescribed_defaults = (
                prescribed if isinstance(prescribed, Mapping) else prescribed(scenario)
            )
            profile_forcing = launch.get("forcing", {})
            forcing = Forcing(
                constant={
                    **prescribed_defaults,
                    **profile_forcing.get("constant", {}),
                    **self.forcing.constant,
                },
                series={**profile_forcing.get("series", {}), **self.forcing.series},
            )
        elif self.scenario:
            raise ValueError("scenario controls require a builtin preset")
        local_x, local_y = self.grid.local_shape(self.execution.partitions)
        conf = Configuration(
            **settings,
            nx=local_x,
            ny=local_y,
            use_sharding=self.execution.partitions != (1, 1),
        )
        constants = {**launch.get("constants", {}), **self.constants}
        if "dtype" in constants and constants["dtype"] != conf.dtype:
            raise ValueError("physical constants dtype must match numerical dtype")
        physical_options: dict[str, Any] = {**constants, "dtype": conf.dtype}
        phys = PhysicalConstants(**physical_options)
        if self.physics == "coupled" and conf.deltatDyn != conf.deltatTherm:
            raise ValueError(
                "coupled physics requires equal dynamics/thermodynamic timesteps"
            )
        object.__setattr__(self, "configuration", conf)
        object.__setattr__(self, "physical_constants", phys)
        object.__setattr__(self, "resolved_scenario", MappingProxyType(scenario))
        object.__setattr__(self, "resolved_forcing", forcing)
        object.__setattr__(
            self,
            "resolved_constants",
            MappingProxyType({name: getattr(phys, name) for name in constants}),
        )
        object.__setattr__(
            self,
            "resolved_settings",
            MappingProxyType({name: getattr(conf, name) for name in settings}),
        )
        initial = _freeze_fields(self.initial_conditions, INITIAL_FIELDS)
        shape = (self.grid.nx, self.grid.ny)
        object.__setattr__(self, "initial_conditions", initial)
        profile_initial = launch.get("initial_conditions", {})
        resolved_initial = (
            _freeze_fields({**profile_initial, **initial}, INITIAL_FIELDS)
            if profile_initial
            else initial
        )
        _validate_shapes(resolved_initial, shape)
        object.__setattr__(self, "resolved_initial_conditions", resolved_initial)
        forcing.validate_shape(shape)
        if forcing.length is not None and self.steps > forcing.length:
            raise ValueError("requested steps exceed forcing series length")
        # Validate the host clock before any resource allocation.
        _ = self.step_duration

    @classmethod
    def builtin(cls, name: str, **options: Any) -> "Experiment":
        """Define a registered scientific preset using its authoritative defaults.

        Grid defaults come from the public Grid registry, except the growth
        column uses its scenario's 2 by 2 grid. Explicit options override preset
        settings; scenario contains only experiment-specific coefficients.
        """
        return cls(preset=name, **options)

    @classmethod
    def from_file(
        cls, path: str | Path, *, overrides: Mapping[str, Any] | None = None
    ) -> "Experiment":
        """Read versioned JSON with registry < file < Python override precedence."""
        from veris.model.experiment_file import read_experiment

        return read_experiment(path, overrides=overrides)

    def to_dict(self, *, mode: str = "resolved") -> dict[str, Any]:
        """Export pinned resolved values or sparse authored controls as fresh JSON data."""
        from veris.model.experiment_file import experiment_to_dict

        return experiment_to_dict(self, mode=mode)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "Experiment":
        """Merge explicit grouped overrides into authored inputs and revalidate.

        Shared defaults < preset < authored configuration < these overrides.
        Nested mappings merge by key; arrays, sequences, None and scalar values
        replace whole values. The preceding definition remains unchanged.
        """
        from veris.model.experiment_file import experiment_from_dict
        from veris.model.experiment_resolution import (
            constructor_inputs,
            merge_overrides,
        )

        return experiment_from_dict(
            merge_overrides(constructor_inputs(self), overrides)
        )

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
