"""Run the standalone fixed-wind dynamics experiment.

The reference jax_halo_exchange initialize_dyn.py prescribes a rotating wind
snapshot over a square basin with two closed walls. Arrays use Veris (x, y)
ordering and two periodic halo cells per partition. Rectangular grids extend
the same equations with independent x/y spacing. No thermodynamic kernel runs.
Scenario coefficients are experimental controls, not universal physical laws.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import click
import numpy as np
from jax.sharding import Mesh

from veris.io.cli import (
    console_main,
    driver_definition,
    output_options,
    parameter_namespace,
    parse_options,
    run_driver,
)
from veris.model.configuration import Configuration
from veris.model.metadata import FROM_REGISTRY, registry_defaults, validate_scalars
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import Parameter, State
from veris.model.variables import VARIABLES

DYNAMICS_SETTINGS = {
    "length": Parameter(
        512000.0, float, "Basin length in each horizontal direction", "m"
    ),
    "wind_time": Parameter(5.875, float, "Fixed reference wind snapshot time", "day"),
    "wind_speed": Parameter(15.0, float, "Wind amplitude", "m s-1"),
    "wind_angle": Parameter(0.4 * np.pi, float, "Wind rotation angle", "rad"),
    "wind_scale": Parameter(50000.0, float, "Wind radial normalization length", "m"),
    "wind_decay": Parameter(100000.0, float, "Wind exponential decay length", "m"),
    "center_offset": Parameter(
        0.1, float, "Initial wind center as basin fraction", "1"
    ),
    "center_rate": Parameter(
        0.1, float, "Wind center translation as basin fraction per day", "day-1"
    ),
    "ocean_speed": Parameter(0.01, float, "Ocean circulation speed scale", "m s-1"),
    "depth": Parameter(1000.0, float, "Ocean basin depth", "m"),
    "ice_thickness": Parameter(
        0.3, float, "Initial ice thickness including wall cells", "m"
    ),
    "surface_temperature": Parameter(
        273.0, float, "Initial ice surface temperature", "K"
    ),
    "coriolis_start": Parameter(
        1.4604e-4, float, "Coriolis frequency at first y cell", "s-1"
    ),
    "coriolis_offset": Parameter(
        1.4596e-4, float, "Reference last-cell Coriolis intercept", "s-1"
    ),
    "coriolis_increment": Parameter(
        8.0e-8, float, "Reference Coriolis increment per y cell", "s-1"
    ),
}


@dataclass(frozen=True)
@registry_defaults(DYNAMICS_SETTINGS)
class DynamicsSettings:
    """Immutable reference scenario controls, separate from differentiable State."""

    length: float = FROM_REGISTRY
    wind_time: float = FROM_REGISTRY
    wind_speed: float = FROM_REGISTRY
    wind_angle: float = FROM_REGISTRY
    wind_scale: float = FROM_REGISTRY
    wind_decay: float = FROM_REGISTRY
    center_offset: float = FROM_REGISTRY
    center_rate: float = FROM_REGISTRY
    ocean_speed: float = FROM_REGISTRY
    depth: float = FROM_REGISTRY
    ice_thickness: float = FROM_REGISTRY
    surface_temperature: float = FROM_REGISTRY
    coriolis_start: float = FROM_REGISTRY
    coriolis_offset: float = FROM_REGISTRY
    coriolis_increment: float = FROM_REGISTRY

    def __post_init__(self) -> None:
        """Reject nonfinite coefficients and invalid geometric denominators."""
        validate_scalars(
            self,
            DYNAMICS_SETTINGS,
            positive=frozenset(
                {"length", "wind_scale", "wind_decay", "depth", "surface_temperature"}
            ),
        )
        if self.ice_thickness < 0:
            raise ValueError("ice_thickness must be nonnegative")


def _fields(
    ix: np.ndarray,
    iy: np.ndarray,
    nx: int,
    ny: int,
    scenario: DynamicsSettings,
    dtype: str,
) -> dict[str, np.ndarray]:
    """Evaluate reference fields at global periodic cell indices on the host."""
    s = scenario
    dx, dy = s.length / (nx - 1), s.length / (ny - 1)
    x, y = np.meshgrid((ix + 0.5) * dx, (iy + 0.5) * dy, indexing="ij")
    i, j = np.meshgrid(ix, iy, indexing="ij")
    mask = ((i != nx - 1) & (j != ny - 1)).astype(dtype)
    west = mask * ((i - 1) % nx != nx - 1)
    south = mask * ((j - 1) % ny != ny - 1)
    center = s.length * (s.center_offset + s.center_rate * s.wind_time)
    wx, wy = x - center, y - center
    scale = -s.wind_speed * np.exp(-np.hypot(wx, wy) / s.wind_decay) / s.wind_scale
    values = {
        "hIceMean": s.ice_thickness,
        "hSnowMean": 0.0,
        "Area": 1.0,
        "TSurf": s.surface_temperature,
        "maskInC": mask,
        "maskInU": west,
        "maskInV": south,
        "iceMask": mask,
        "iceMaskU": west,
        "iceMaskV": south,
        "uWind": scale * (np.cos(s.wind_angle) * wx + np.sin(s.wind_angle) * wy),
        "vWind": scale * (-np.sin(s.wind_angle) * wx + np.cos(s.wind_angle) * wy),
        "uOcean": s.ocean_speed * (2 * y - s.length) / s.length * west,
        "vOcean": -s.ocean_speed * (2 * x - s.length) / s.length * south,
        "R_low": -s.depth * mask,
        "fCori": s.coriolis_start
        + j
        / (ny - 1)
        * (s.coriolis_offset + s.coriolis_increment * ny - s.coriolis_start),
        "rAz": dx * dy,
        "recip_rA": 1 / (dx * dy),
        "recip_rAu": 1 / (dx * dy),
        "recip_rAv": 1 / (dx * dy),
    }
    for name in ("dxG", "dxU", "dxV"):
        values[name] = dx
    for name in ("dyG", "dyU", "dyV"):
        values[name] = dy
    for name in ("recip_dxC", "recip_dxU", "recip_dxV"):
        values[name] = 1 / dx
    for name in ("recip_dyC", "recip_dyU", "recip_dyV"):
        values[name] = 1 / dy
    return {
        name: np.full(x.shape, values.get(name, 0.0), dtype=dtype) for name in VARIABLES
    }


DYNAMICS_NUMERICAL_DEFAULTS: dict[str, Any] = {
    "deltatDyn": 600.0,
    "deltatTherm": 600.0,
    "useEVP": True,
    "useFreedrift": False,
    "useAdaptiveEVP": True,
    "useRelativeWind": False,
    "evpAlpha": 500.0,
    "evpBeta": 500.0,
    "nEVPsteps": 120,
}
"""Reference solver choices for dynamics experiment definitions."""

DYNAMICS_DEFAULT_EXTENT = 1024
DYNAMICS_DEFAULT_STEPS = 100
DYNAMICS_PROFILES: dict[str, Any] = {
    "example": {"grid": (16, 24), "steps": DYNAMICS_DEFAULT_STEPS},
    "standalone": {
        "grid": (DYNAMICS_DEFAULT_EXTENT,) * 2,
        "steps": DYNAMICS_DEFAULT_STEPS,
        "execution": {"backend": "cpu"},
    },
    "parallel": {
        "grid": (DYNAMICS_DEFAULT_EXTENT,) * 2,
        "steps": 1000,
        "execution": {"backend": "cpu", "all_devices": True},
    },
}
"""Intentional driver/example grids and durations with explicit owners."""


def build_fields(
    state: State,
    configuration: Configuration,
    constants: PhysicalConstants,
    scenario: Mapping[str, Any],
    global_shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    *,
    regional: bool,
) -> dict[str, np.ndarray]:
    """Supply all notebook startup fields at global physical indices."""
    return _fields(
        x, y, *global_shape, DynamicsSettings(**scenario), configuration.dtype
    )


def initialize(
    nx: int = DYNAMICS_DEFAULT_EXTENT,
    ny: int | None = None,
    *,
    mesh: Mesh | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
    scenario_overrides: Mapping[str, Any] | None = None,
    physical_overrides: Mapping[str, Any] | None = None,
) -> tuple[State, Configuration, PhysicalConstants]:
    """Initialize global interior dimensions with local halos on each mesh shard.

    Only addressable partitions are materialized on each process. This supports
    multi-process CPU execution without allocating a full global state per rank.
    Returned configuration nx/ny are local interior extents. Wind stays fixed
    throughout integration, matching experiment snapshot 15 of the shifted field.
    """
    from veris.setups.presets import _initialize_adapter

    return _initialize_adapter(
        "dynamics",
        nx=nx,
        ny=nx if ny is None else ny,
        mesh=mesh,
        settings=settings_overrides,
        constants=physical_overrides,
        scenario=scenario_overrides,
    )


@click.command(help=__doc__)
@click.option("--steps", type=click.IntRange(min=0))
@click.option("--backend", type=click.Choice(["cpu", "gpu"]))
@click.option("--nx", type=click.IntRange(min=2))
@click.option("--ny", type=click.IntRange(min=2))
@click.option("--evp-steps", type=click.IntRange(min=1))
@click.option(
    "--set", "set_values", multiple=True, help="Grouped override: SECTION.KEY=JSON."
)
@output_options("dynamics.nc", tuple(VARIABLES))
def cli(**kwargs: Any) -> SimpleNamespace:
    """Parse explicit controls without promoting parser defaults to overrides."""
    return parameter_namespace(kwargs)


def main(argv: list[str] | None = None) -> None:
    """Translate standalone profile flags and invoke the public Python runner."""
    args = parse_options(cli, argv)
    definition = driver_definition("dynamics", "standalone", args)
    try:
        result = run_driver(definition, args)
    except ValueError as error:
        if str(error).startswith("nonfinite output"):
            raise RuntimeError(str(error)) from error
        raise
    assert result.execution_metadata is not None
    if result.execution_metadata["process_index"] == 0:
        print(
            f"dynamics: {result.completed_steps} steps, {result.execution_metadata['backend']}, compile {result.compilation_seconds:.3f}s, run {result.integration_seconds:.3f}s, {args.output}"
        )


if __name__ == "__main__":
    console_main(main)
