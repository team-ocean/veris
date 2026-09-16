"""Standalone dynamics and growth on a periodic Cartesian sea with an island.

The small example follows the jax_halo_exchange reference integration sequence.
It uses prescribed ocean/atmospheric fields, not an evolving ocean or geographic
coastline. Fields are (nx + 4, ny + 4) JAX arrays with two periodic halo cells.
The host driver selects serial halos; state is an immutable dataclass PyTree.
Five EVP substeps keep this example small; they are not a convergence criterion.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from veris._metadata import (
    FROM_REGISTRY,
    registry_defaults,
    validate_scalars,
)
from veris._typing import Parameter, State, jit
from veris.configuration import SETTINGS, Configuration
from veris.diagnostics import Diagnostics
from veris.initialization import initialize as initialize_model
from veris.physical_constants import PhysicalConstants

ARTIFICIAL_SETTINGS: dict[str, Parameter] = {
    "saltOcn_ref": Parameter(
        34.7, float, "Prescribed ocean salinity in the artificial example", "g kg-1"
    ),
    "artificialGridSpacing": Parameter(
        8000.0, float, "Uniform Cartesian grid spacing in the artificial example", "m"
    ),
    "artificialWindSpeed": Parameter(
        5.0, float, "Prescribed signed zonal wind in the artificial example", "m s-1"
    ),
    "artificialAirTemperature": Parameter(
        260.0,
        float,
        "Prescribed atmosphere and initial ice-surface temperature in the artificial example",
        "K",
    ),
    "artificialIceThickness": Parameter(
        1.0,
        float,
        "Initial grid-cell mean ice thickness over ocean in the artificial example",
        "m",
    ),
    "artificialSnowThickness": Parameter(
        0.05,
        float,
        "Initial grid-cell mean snow thickness over ocean in the artificial example",
        "m",
    ),
    "artificialIceArea": Parameter(
        0.8,
        float,
        "Initial ocean-cell ice concentration in the artificial example",
        "1",
    ),
    "artificialOceanDepth": Parameter(
        -100.0, float, "Signed ocean bottom elevation in the artificial example", "m"
    ),
    "artificialCoriolis": Parameter(
        0.0001, float, "Uniform Coriolis frequency in the artificial example", "s-1"
    ),
    "artificialCooling": Parameter(
        100.0,
        float,
        "Default upward open-water cooling imposed each artificial step",
        "W m-2",
    ),
    "artificialTimeStep": Parameter(
        600.0,
        float,
        "Default dynamics and thermodynamics timestep for the artificial example",
        "s",
    ),
    "artificialEVPsteps": Parameter(
        5, int, "Default EVP substeps in the artificial example", "1"
    ),
}


@dataclass(frozen=True)
@registry_defaults({"dtype": SETTINGS["dtype"], **ARTIFICIAL_SETTINGS})
class ArtificialSettings:
    """Validated scenario defaults kept outside model configuration and AD State."""

    dtype: str = field(default=FROM_REGISTRY, kw_only=True)

    saltOcn_ref: float = FROM_REGISTRY
    artificialGridSpacing: float = FROM_REGISTRY
    artificialWindSpeed: float = FROM_REGISTRY
    artificialAirTemperature: float = FROM_REGISTRY
    artificialIceThickness: float = FROM_REGISTRY
    artificialSnowThickness: float = FROM_REGISTRY
    artificialIceArea: float = FROM_REGISTRY
    artificialOceanDepth: float = FROM_REGISTRY
    artificialCoriolis: float = FROM_REGISTRY
    artificialCooling: float = FROM_REGISTRY
    artificialTimeStep: float = FROM_REGISTRY
    artificialEVPsteps: int = FROM_REGISTRY

    def __post_init__(self) -> None:
        """Validate prescribed experiment fields at the selected model precision."""
        validate_scalars(
            self,
            ARTIFICIAL_SETTINGS,
            positive=frozenset(
                {
                    "artificialGridSpacing",
                    "artificialAirTemperature",
                    "artificialTimeStep",
                    "artificialEVPsteps",
                }
            ),
        )
        for name in ("artificialIceThickness", "artificialSnowThickness"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.artificialIceArea <= 1:
            raise ValueError("artificialIceArea must lie between zero and one")
        if self.artificialOceanDepth > 0:
            raise ValueError("artificialOceanDepth must be nonpositive")


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
    """Return an artificial island experiment with separate local scenario controls.

    Registry defaults select an 8-km grid, 600-second timesteps and five EVP
    substeps. Explicit arguments override scenario settings and are recorded in
    the initialized arrays; explicit deltatDyn, deltatTherm and nEVPsteps overrides
    take precedence over the artificial scenario defaults. Atmosphere is
    saturated at its specified temperature with blackbody downward longwave
    radiation. This setup supports serial execution; use the general initializer
    with supplied mesh geometry for sharded experiments. Scenario overrides stay
    local to this module; use the step cooling argument for heat-flux forcing.
    """
    overrides = dict(settings_overrides or {})
    scenario_values = dict(scenario_overrides or {})
    if "artificialCooling" in scenario_values:
        raise ValueError("pass artificialCooling as the cooling argument to step")
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
        ("artificialWindSpeed", wind),
        ("artificialAirTemperature", air_temperature),
    ):
        if value is not None:
            scenario_values[name] = value
    scenario = ArtificialSettings(dtype=controls.dtype, **scenario_values)
    if "use_sharding" in overrides and controls.use_sharding:
        raise ValueError("artificial initialization supports serial execution only")
    if controls.nx < 4 or controls.ny < 4:
        raise ValueError("grid dimensions must be at least four interior cells")
    overrides["use_sharding"] = False
    overrides.setdefault("deltatTherm", scenario.artificialTimeStep)
    overrides.setdefault("deltatDyn", scenario.artificialTimeStep)
    overrides.setdefault("nEVPsteps", scenario.artificialEVPsteps)
    vs, conf, phys = initialize_model(
        settings_overrides=overrides, physical_overrides=physical_overrides
    )
    nx, ny = conf.nx, conf.ny
    wind = scenario.artificialWindSpeed
    air_temperature = scenario.artificialAirTemperature
    ones = jnp.ones_like(vs.iceMask)
    fields = {}
    interior = np.ones((nx, ny))
    interior[nx // 2 - 1 : nx // 2 + 1, ny // 2 - 1 : ny // 2 + 1] = 0
    mask = jnp.asarray(np.pad(interior, 2, mode="wrap"), dtype=vs.iceMask.dtype)
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
    spacing = scenario.artificialGridSpacing
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
        hIceMean=scenario.artificialIceThickness * mask,
        hSnowMean=scenario.artificialSnowThickness * mask,
        Area=scenario.artificialIceArea * mask,
        TSurf=air_temperature * ones,
        SeaIceLoad=(
            scenario.artificialIceThickness * phys.rhoIce
            + scenario.artificialSnowThickness * phys.rhoSnow
        )
        * mask,
        recip_hIceMean=1
        / jnp.sqrt((scenario.artificialIceThickness * mask) ** 2 + phys.hIce_reg),
        R_low=scenario.artificialOceanDepth * ones,
        fCori=scenario.artificialCoriolis * ones,
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
    vs = replace(vs, **fields)
    if not conf.enable_cyclic_y:
        from veris.fill_overlap import fill_state_overlap

        vs = fill_state_overlap(vs, conf)
    return vs, conf, phys


def step(
    vs: State,
    conf: Configuration,
    phys: PhysicalConstants,
    cooling: float | jax.Array | None = None,
) -> State:
    """Advance the calculation state with prescribed upward open-water cooling.

    The Python driver and compiled_step execute the same dynamics, transport,
    cleanup and growth sequence. Both support AD. Use step_with_diagnostics to
    retain the output-only ice-ocean coupling fields alongside the updated state.
    Omitted cooling uses ARTIFICIAL_SETTINGS; pass cooling explicitly to change
    the forcing. Cooling resets atmospheric Qnet/Qsw forcing on every call, before Growth
    replaces these state fields with ocean-coupling fluxes.
    """
    result, _ = step_with_diagnostics(vs, conf, phys, cooling)
    return result


def step_with_diagnostics(
    vs: State,
    conf: Configuration,
    phys: PhysicalConstants,
    cooling: float | jax.Array | None = None,
) -> tuple[State, Diagnostics]:
    """Advance a coupled step and return separate periodic ocean-coupling outputs.

    Ocean stress is evaluated after momentum and before transport and growth,
    matching the reference integration order. Growth supplies freshwater, salt
    and penetrating shortwave outputs. Only calculation fields enter State.
    """
    if cooling is None:
        cooling = float(cast(float, ARTIFICIAL_SETTINGS["artificialCooling"].default))
    if conf.use_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        if set(mesh.axis_names) != {"x", "y"}:
            raise ValueError(
                "sharded stepping requires an active mesh with axes x and y"
            )
        if not {"x", "y"} <= set(mesh.manual_axes):
            # Run stencils on each local halo-inclusive partition. Explicitly
            # sharded global arrays cannot be rolled along partitioned axes.
            mapped = jax.shard_map(
                lambda state, flux: _step_local(state, conf, phys, flux),
                mesh=mesh,
                in_specs=(P("x", "y"), P()),
                out_specs=(P("x", "y"), P("x", "y")),
            )
            return mapped(vs, cooling)
    return _step_local(vs, conf, phys, cooling)


def _step_local(
    vs: State, conf: Configuration, phys: PhysicalConstants, cooling: float | jax.Array
) -> tuple[State, Diagnostics]:
    """Execute the reference physics sequence on one local halo-inclusive grid."""
    from veris.advection import Advection
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.clean_up import clean_up_advection, ridging
    from veris.dynamics_routines import SeaIceStrength
    from veris.dynsolver import IceVelocities, WindForcingXY
    from veris.fill_overlap import fill_state_overlap
    from veris.growth import Growth
    from veris.ocean_stress import OceanStressUV

    def assign(state: State, names: str, values: tuple[jax.Array, ...]) -> State:
        return replace(state, **dict(zip(names.split(), values, strict=True)))

    vs = replace(vs, Qnet=jnp.full_like(vs.Qnet, cooling), Qsw=jnp.zeros_like(vs.Qsw))
    vs = assign(vs, "SeaIceMassC SeaIceMassU SeaIceMassV", SeaIceMass(vs, conf, phys))
    vs = assign(vs, "AreaW AreaS", AreaWS(vs, conf, phys))
    vs = assign(vs, "WindForcingX WindForcingY", WindForcingXY(vs, conf, phys))
    vs = replace(vs, SeaIceStrength=SeaIceStrength(vs, conf, phys))
    vs = assign(
        vs,
        "uIce vIce sigma1 sigma2 sigma12",
        IceVelocities(
            vs, conf, phys, axis_names=("x", "y") if conf.use_sharding else ()
        ),
    )
    ocean_stress_u, ocean_stress_v = OceanStressUV(vs, conf, phys)
    vs = assign(vs, "hIceMean hSnowMean Area", Advection(vs, conf, phys))
    vs = assign(
        vs,
        "hIceMean hSnowMean Area TSurf os_hIceMean os_hSnowMean",
        clean_up_advection(vs, conf, phys),
    )
    vs = replace(vs, Area=ridging(vs, conf, phys))
    (
        ice,
        snow,
        area,
        temperature,
        freshwater,
        salt,
        shortwave,
        net_heat,
        load,
        penetrating_shortwave,
        inverse_ice,
    ) = Growth(vs, conf, phys)
    vs = replace(
        vs,
        hIceMean=ice,
        hSnowMean=snow,
        Area=area,
        TSurf=temperature,
        Qsw=shortwave,
        Qnet=net_heat,
        SeaIceLoad=load,
        recip_hIceMean=inverse_ice,
    )
    diagnostics = Diagnostics(
        IcePenetSW=penetrating_shortwave,
        OceanStressU=ocean_stress_u,
        OceanStressV=ocean_stress_v,
        EmPmR=freshwater,
        forc_salt_surface=salt,
    )
    return fill_state_overlap(vs, conf), fill_state_overlap(diagnostics, conf)


compiled_step = jit(step, static_argnames=["conf", "phys"])
"""Whole-step compiled driver; settings and constants are static; cooling stays dynamic.

Shares the exact physics sequence with step. Choose this callable once outside
the integration loop for workloads where reduced host dispatch is beneficial;
see benchmarks/README.md for CPU/GPU measurements and their limits.
"""
