"""Standalone dynamics and growth on a periodic Cartesian sea with an island.

The small example follows the jax_halo_exchange reference integration sequence.
It uses prescribed ocean/atmospheric fields, not an evolving ocean or geographic
coastline. Fields are (nx + 4, ny + 4) JAX arrays with two periodic halo cells.
The host driver selects serial halos; state is an immutable dataclass PyTree.
Five EVP substeps keep this example small; they are not a convergence criterion.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from veris.execution.initialization import initialize as initialize_model
from veris.model.configuration import SETTINGS, Configuration
from veris.model.metadata import (
    FROM_REGISTRY,
    registry_defaults,
    validate_scalars,
)
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import Parameter, State

ISLAND_SETTINGS: dict[str, Parameter] = {
    "saltOcn_ref": Parameter(
        34.7, float, "Prescribed ocean salinity in the island example", "g kg-1"
    ),
    "islandGridSpacing": Parameter(
        8000.0, float, "Uniform Cartesian grid spacing in the island example", "m"
    ),
    "islandWindSpeed": Parameter(
        5.0, float, "Prescribed signed zonal wind in the island example", "m s-1"
    ),
    "islandAirTemperature": Parameter(
        260.0,
        float,
        "Prescribed atmosphere and initial ice-surface temperature in the island example",
        "K",
    ),
    "islandIceThickness": Parameter(
        1.0,
        float,
        "Initial grid-cell mean ice thickness over ocean in the island example",
        "m",
    ),
    "islandSnowThickness": Parameter(
        0.05,
        float,
        "Initial grid-cell mean snow thickness over ocean in the island example",
        "m",
    ),
    "islandIceArea": Parameter(
        0.8,
        float,
        "Initial ocean-cell ice concentration in the island example",
        "1",
    ),
    "islandOceanDepth": Parameter(
        -100.0, float, "Signed ocean bottom elevation in the island example", "m"
    ),
    "islandCoriolis": Parameter(
        0.0001, float, "Uniform Coriolis frequency in the island example", "s-1"
    ),
    "islandCooling": Parameter(
        100.0,
        float,
        "Default upward open-water cooling imposed each island step",
        "W m-2",
    ),
    "islandTimeStep": Parameter(
        600.0,
        float,
        "Default dynamics and thermodynamics timestep for the island example",
        "s",
    ),
    "islandEVPsteps": Parameter(
        5, int, "Default EVP substeps in the island example", "1"
    ),
}


@dataclass(frozen=True)
@registry_defaults({"dtype": SETTINGS["dtype"], **ISLAND_SETTINGS})
class IslandSettings:
    """Validated scenario defaults kept outside model configuration and AD State."""

    dtype: str = field(default=FROM_REGISTRY, kw_only=True)

    saltOcn_ref: float = FROM_REGISTRY
    islandGridSpacing: float = FROM_REGISTRY
    islandWindSpeed: float = FROM_REGISTRY
    islandAirTemperature: float = FROM_REGISTRY
    islandIceThickness: float = FROM_REGISTRY
    islandSnowThickness: float = FROM_REGISTRY
    islandIceArea: float = FROM_REGISTRY
    islandOceanDepth: float = FROM_REGISTRY
    islandCoriolis: float = FROM_REGISTRY
    islandCooling: float = FROM_REGISTRY
    islandTimeStep: float = FROM_REGISTRY
    islandEVPsteps: int = FROM_REGISTRY

    def __post_init__(self) -> None:
        """Validate prescribed experiment fields at the selected model precision."""
        validate_scalars(
            self,
            ISLAND_SETTINGS,
            positive=frozenset(
                {
                    "islandGridSpacing",
                    "islandAirTemperature",
                    "islandTimeStep",
                    "islandEVPsteps",
                }
            ),
        )
        for name in ("islandIceThickness", "islandSnowThickness"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.islandIceArea <= 1:
            raise ValueError("islandIceArea must lie between zero and one")
        if self.islandOceanDepth > 0:
            raise ValueError("islandOceanDepth must be nonpositive")


def numerical_defaults(scenario: IslandSettings) -> dict[str, Any]:
    """Return the scenario's numerical defaults used by experiment initialization."""
    return {
        "deltatTherm": scenario.islandTimeStep,
        "deltatDyn": scenario.islandTimeStep,
        "nEVPsteps": scenario.islandEVPsteps,
    }


def initialize(
    nx: int | None = None,
    ny: int | None = None,
    wind: float | None = None,
    air_temperature: float | None = None,
    *,
    dtype: str | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
    scenario_overrides: Mapping[str, Any] | None = None,
    physical_overrides: Mapping[str, Any] | None = None,
) -> tuple[State, Configuration, PhysicalConstants]:
    """Return an island experiment with separate local scenario controls.

    Registry defaults select an 8-km grid, 600-second timesteps and five EVP
    substeps. Explicit arguments override scenario settings and are recorded in
    the initialized arrays; explicit deltatDyn, deltatTherm and nEVPsteps overrides
    take precedence over the island scenario defaults. Atmosphere is
    saturated at its specified temperature with blackbody downward longwave
    radiation. This setup supports serial execution; use the general initializer
    with supplied mesh geometry for sharded experiments. Scenario overrides stay
    local to this module; prescribe heat flux through Experiment forcing.
    """
    overrides = dict(settings_overrides or {})
    scenario_values = dict(scenario_overrides or {})
    if "islandCooling" in scenario_values:
        raise ValueError("prescribe islandCooling through Experiment forcing")
    for name, value in (
        ("nx", nx),
        ("ny", ny),
    ):
        if value is not None:
            overrides[name] = value
    if dtype is not None:
        overrides["dtype"] = dtype
    controls = Configuration(**overrides)
    for name, value in (
        ("islandWindSpeed", wind),
        ("islandAirTemperature", air_temperature),
    ):
        if value is not None:
            scenario_values[name] = value
    scenario = IslandSettings(dtype=controls.dtype, **scenario_values)
    if "use_sharding" in overrides and controls.use_sharding:
        raise ValueError("island initialization supports serial execution only")
    if controls.nx < 4 or controls.ny < 4:
        raise ValueError("grid dimensions must be at least four interior cells")
    overrides["use_sharding"] = False
    for name, value in numerical_defaults(scenario).items():
        overrides.setdefault(name, value)
    vs, conf, phys = initialize_model(
        settings_overrides=overrides, physical_overrides=physical_overrides
    )
    fields = _fields(
        vs,
        phys,
        scenario,
        conf.nx,
        conf.ny,
        (np.arange(conf.nx + 4) - 2) % conf.nx,
        (np.arange(conf.ny + 4) - 2) % conf.ny,
    )
    vs = replace(vs, **fields)
    if not conf.enable_cyclic_y:
        from veris.grid.halos import fill_state_overlap

        vs = fill_state_overlap(vs, conf)
    return vs, conf, phys


def _fields(
    vs: State,
    phys: PhysicalConstants,
    scenario: IslandSettings,
    nx: int,
    ny: int,
    ix: np.ndarray,
    iy: np.ndarray,
    *,
    regional: bool = False,
) -> dict[str, jax.Array]:
    """Evaluate exact startup arithmetic at global physical cell indices.

    Regional allocation reuses these expressions, so a central island stays in
    the global basin even when a partition has only two interior cells.
    """
    wind = scenario.islandWindSpeed
    air_temperature = scenario.islandAirTemperature
    ones = jnp.ones_like(vs.iceMask)
    fields = {}
    land_x = (ix >= nx // 2 - 1) & (ix < nx // 2 + 1)
    land_y = (iy >= ny // 2 - 1) & (iy < ny // 2 + 1)
    mask = jnp.asarray(~(land_x[:, None] & land_y[None, :]), dtype=vs.iceMask.dtype)
    previous_x = (ix - 1) % nx
    previous_y = (iy - 1) % ny
    west_land = (previous_x >= nx // 2 - 1) & (previous_x < nx // 2 + 1)
    south_land = (previous_y >= ny // 2 - 1) & (previous_y < ny // 2 + 1)
    west = mask * jnp.asarray(~(west_land[:, None] & land_y[None, :]), dtype=mask.dtype)
    south = mask * jnp.asarray(
        ~(land_x[:, None] & south_land[None, :]), dtype=mask.dtype
    )
    if not regional:
        # Preserve the original unrefreshed outermost serial mask halos in
        # step-zero reference data. Regional boundaries instead use global
        # predecessor cells, never a wrap within one addressable partition.
        west = mask * jnp.roll(mask, 1, axis=0)
        south = mask * jnp.roll(mask, 1, axis=1)
    fields.update(
        iceMask=mask,
        maskInC=mask,
        iceMaskU=west,
        maskInU=west,
        iceMaskV=south,
        maskInV=south,
    )
    # Direct metrics unused by the kernels stay local to initialization.
    spacing = scenario.islandGridSpacing
    cell_area = spacing**2
    for name in ("dxG", "dyG", "dxU", "dyU", "dxV", "dyV"):
        fields[name] = spacing * ones
    for name in ("dxC", "dyC", "dxU", "dyU", "dxV", "dyV"):
        fields["recip_" + name] = ones / spacing
    fields["rAz"] = cell_area * ones
    for name in ("rA", "rAu", "rAv"):
        fields["recip_" + name] = ones / cell_area
    temperature = phys.celsius2K + phys.tempFrz
    vapor = 10 ** (
        phys.iceVaporPressureLog10Offset
        - phys.iceVaporPressureTemperature / air_temperature
    )
    fields.update(
        hIceMean=scenario.islandIceThickness * mask,
        hSnowMean=scenario.islandSnowThickness * mask,
        Area=scenario.islandIceArea * mask,
        TSurf=air_temperature * ones,
        SeaIceLoad=(
            scenario.islandIceThickness * phys.rhoIce
            + scenario.islandSnowThickness * phys.rhoSnow
        )
        * mask,
        recip_hIceMean=1
        / jnp.sqrt((scenario.islandIceThickness * mask) ** 2 + phys.hIce_reg),
        R_low=scenario.islandOceanDepth * ones,
        fCori=scenario.islandCoriolis * ones,
        theta=temperature * ones,
        ocSalt=scenario.saltOcn_ref * ones,
        uWind=wind * ones,
        wSpeed=abs(wind) * ones,
        ATemp=air_temperature * ones,
        LWdown=phys.stefBoltz * air_temperature**4 * ones,
        aqh=phys.waterVaporDryAirMassRatio
        * vapor
        / (phys.iceSurfacePressure - (1 - phys.waterVaporDryAirMassRatio) * vapor)
        * ones,
    )
    return fields
