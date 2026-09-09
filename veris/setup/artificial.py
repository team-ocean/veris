"""Standalone dynamics and growth on a periodic Cartesian sea with an island.

The small example follows the jax_halo_exchange reference integration sequence.
It uses prescribed ocean/atmospheric fields, not an evolving ocean or geographic
coastline. Fields are (nx + 4, ny + 4) JAX arrays with two periodic halo cells.
The host driver selects serial halos; state is an immutable dataclass PyTree.
Five EVP substeps keep this example small; they are not a convergence criterion.
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from veris._typing import jit
from veris.configuration import Settings
from veris.diagnostics import Diagnostics
from veris.initialization import initialize as initialize_model
from veris.physical_constants import PhysicalConstants
from veris.state import State


def initialize(
    nx: int | None = None,
    ny: int | None = None,
    wind: float | None = None,
    air_temperature: float | None = None,
    *,
    settings_overrides: Mapping[str, Any] | None = None,
    physical_overrides: Mapping[str, Any] | None = None,
) -> tuple[State, Settings, PhysicalConstants]:
    """Return an artificial island experiment with all controls in Settings.

    Registry defaults select an 8-km grid, 600-second timesteps and five EVP
    substeps. Explicit arguments override scenario settings and are recorded in
    the returned object; explicit deltatDyn, deltatTherm and nEVPsteps overrides
    take precedence over the artificial scenario defaults. Atmosphere is
    saturated at its specified temperature with blackbody downward longwave
    radiation. This setup supports serial execution; use the general initializer
    with supplied mesh geometry for sharded experiments.
    """
    overrides = dict(settings_overrides or {})
    for name, value in (
        ("nx", nx),
        ("ny", ny),
        ("artificialWindSpeed", wind),
        ("artificialAirTemperature", air_temperature),
    ):
        if value is not None:
            overrides[name] = value
    controls = Settings(**overrides)
    if "use_sharding" in overrides and controls.use_sharding:
        raise ValueError("artificial initialization supports serial execution only")
    if controls.nx < 4 or controls.ny < 4:
        raise ValueError("grid dimensions must be at least four interior cells")
    overrides["use_sharding"] = False
    overrides.setdefault("deltatTherm", controls.artificialTimeStep)
    overrides.setdefault("deltatDyn", controls.artificialTimeStep)
    overrides.setdefault("nEVPsteps", controls.artificialEVPsteps)
    vs, sett, phys = initialize_model(
        settings_overrides=overrides, physical_overrides=physical_overrides
    )
    nx, ny = sett.nx, sett.ny
    wind = sett.artificialWindSpeed
    air_temperature = sett.artificialAirTemperature
    ones = jnp.ones_like(vs.iceMask)
    fields = {}
    interior = np.ones((nx, ny))
    interior[nx // 2 - 1 : nx // 2 + 1, ny // 2 - 1 : ny // 2 + 1] = 0
    mask = jnp.asarray(np.pad(interior, 2, mode="wrap"))
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
    spacing = sett.artificialGridSpacing
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
        hIceMean=sett.artificialIceThickness * mask,
        hSnowMean=sett.artificialSnowThickness * mask,
        Area=sett.artificialIceArea * mask,
        TSurf=air_temperature * ones,
        SeaIceLoad=(
            sett.artificialIceThickness * phys.rhoIce
            + sett.artificialSnowThickness * phys.rhoSnow
        )
        * mask,
        recip_hIceMean=1
        / jnp.sqrt((sett.artificialIceThickness * mask) ** 2 + sett.hIce_reg),
        R_low=sett.artificialOceanDepth * ones,
        fCori=sett.artificialCoriolis * ones,
        theta=temperature * ones,
        ocSalt=phys.saltOcn_ref * ones,
        uWind=wind * ones,
        wSpeed=abs(wind) * ones,
        ATemp=air_temperature * ones,
        LWdown=phys.stefBoltz * air_temperature**4 * ones,
        aqh=phys.waterVaporDryAirMassRatio
        * vapor
        / (phys.iceSurfacePressure - (1 - phys.waterVaporDryAirMassRatio) * vapor)
        * ones,
    )
    return replace(vs, **fields), sett, phys


def step(
    vs: State,
    sett: Settings,
    phys: PhysicalConstants,
    cooling: float | jax.Array | None = None,
) -> State:
    """Advance the calculation state with prescribed upward open-water cooling.

    The Python driver and compiled_step execute the same dynamics, transport,
    cleanup and growth sequence. Both support AD. Use step_with_diagnostics to
    retain the output-only ice-ocean coupling fields alongside the updated state.
    Cooling resets atmospheric Qnet/Qsw forcing on every call, before Growth
    replaces these state fields with ocean-coupling fluxes.
    """
    result, _ = step_with_diagnostics(vs, sett, phys, cooling)
    return result


def step_with_diagnostics(
    vs: State,
    sett: Settings,
    phys: PhysicalConstants,
    cooling: float | jax.Array | None = None,
) -> tuple[State, Diagnostics]:
    """Advance a coupled step and return separate periodic ocean-coupling outputs.

    Ocean stress is evaluated after momentum and before transport and growth,
    matching the reference integration order. Growth supplies freshwater, salt
    and penetrating shortwave outputs. Only calculation fields enter State.
    """
    if cooling is None:
        cooling = sett.artificialCooling
    if sett.use_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        if set(mesh.axis_names) != {"x", "y"}:
            raise ValueError(
                "sharded stepping requires an active mesh with axes x and y"
            )
        if not {"x", "y"} <= set(mesh.manual_axes):
            # Run stencils on each local halo-inclusive partition. Explicitly
            # sharded global arrays cannot be rolled along partitioned axes.
            mapped = jax.shard_map(
                lambda state, flux: _step_local(state, sett, phys, flux),
                mesh=mesh,
                in_specs=(P("x", "y"), P()),
                out_specs=(P("x", "y"), P("x", "y")),
            )
            return mapped(vs, cooling)
    return _step_local(vs, sett, phys, cooling)


def _step_local(
    vs: State, sett: Settings, phys: PhysicalConstants, cooling: float | jax.Array
) -> tuple[State, Diagnostics]:
    """Execute the reference physics sequence on one local halo-inclusive grid."""
    from veris.advection import Advection
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.clean_up import clean_up_advection, ridging
    from veris.dynamics_routines import SeaIceStrength
    from veris.dynsolver import IceVelocities, WindForcingXY
    from veris.fill_overlap import fill_overlap
    from veris.growth import Growth
    from veris.ocean_stress import OceanStressUV

    def assign(state: State, names: str, values: tuple[jax.Array, ...]) -> State:
        return replace(state, **dict(zip(names.split(), values, strict=True)))

    if cooling is None:
        cooling = sett.artificialCooling
    vs = replace(vs, Qnet=jnp.full_like(vs.Qnet, cooling), Qsw=jnp.zeros_like(vs.Qsw))
    vs = assign(vs, "SeaIceMassC SeaIceMassU SeaIceMassV", SeaIceMass(vs, sett, phys))
    vs = assign(vs, "AreaW AreaS", AreaWS(vs, sett, phys))
    vs = assign(vs, "WindForcingX WindForcingY", WindForcingXY(vs, sett, phys))
    vs = replace(vs, SeaIceStrength=SeaIceStrength(vs, sett, phys))
    vs = assign(
        vs,
        "uIce vIce sigma1 sigma2 sigma12",
        IceVelocities(
            vs, sett, phys, axis_names=("x", "y") if sett.use_sharding else ()
        ),
    )
    ocean_stress_u, ocean_stress_v = OceanStressUV(vs, sett, phys)
    vs = assign(vs, "hIceMean hSnowMean Area", Advection(vs, sett, phys))
    vs = assign(
        vs,
        "hIceMean hSnowMean Area TSurf os_hIceMean os_hSnowMean",
        clean_up_advection(vs, sett, phys),
    )
    vs = replace(vs, Area=ridging(vs, sett, phys))
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
    ) = Growth(vs, sett, phys)
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
    return jax.tree.map(lambda array: fill_overlap(array, sett), (vs, diagnostics))


compiled_step = jit(step, static_argnames=["sett", "phys"])
"""Whole-step compiled driver; settings and constants are static; cooling stays dynamic.

Shares the exact physics sequence with step. Choose this callable once outside
the integration loop for workloads where reduced host dispatch is beneficial;
see benchmarks/README.md for CPU/GPU measurements and their limits.
"""
