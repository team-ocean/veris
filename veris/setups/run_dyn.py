"""Fixed-wind dynamics experiment from the standalone run_dyn notebook.

The reference jax_halo_exchange initialize_dyn.py prescribes a rotating wind
snapshot over a square basin with two closed walls. Arrays use Veris (x, y)
ordering and two periodic halo cells per partition. Rectangular grids extend
the same equations with independent x/y spacing. No thermodynamic kernel runs.
Scenario coefficients are experimental controls, not universal physical laws.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from typing import Any

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from veris._metadata import FROM_REGISTRY, registry_defaults, validate_scalars
from veris._typing import Parameter, State
from veris.configuration import Configuration
from veris.diagnostics import Diagnostics
from veris.initialization import initialize as initialize_model
from veris.integration_output import output_callbacks, run_timed
from veris.io.cli import make_output, output_options, parse_options, save_final
from veris.physical_constants import PhysicalConstants
from veris.variables import VARIABLES

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
        name: np.full(x.shape, values.get(name, metadata.default), dtype=dtype)
        for name, metadata in VARIABLES.items()
    }


def initialize(
    nx: int = 1024,
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
    throughout integration, matching notebook snapshot 15 of the shifted field.
    """
    ny = nx if ny is None else ny
    for name, size in [("nx", nx), ("ny", ny)]:
        if type(size) is not int or size < 2:
            raise ValueError(f"{name} must be an integer of at least two cells")
    if mesh is not None and set(mesh.axis_names) != {"x", "y"}:
        raise ValueError("mesh must have axes x and y")
    px, py = (1, 1) if mesh is None else (mesh.shape["x"], mesh.shape["y"])
    if nx % px or ny % py or nx // px < 2 or ny // py < 2:
        raise ValueError(
            "grid must divide evenly into mesh with at least two cells per partition"
        )
    options: dict[str, Any] = {
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
    options.update(settings_overrides or {})
    options.update(nx=nx // px, ny=ny // py, use_sharding=mesh is not None)
    conf = Configuration(**options)
    scenario = DynamicsSettings(**dict(scenario_overrides or {}))
    if jax.dtypes.canonicalize_dtype(conf.dtype) != jnp.dtype(conf.dtype):
        raise ValueError(
            f"dtype {conf.dtype} requires jax_enable_x64 before initialization"
        )
    if mesh is None:
        values = _fields(
            (np.arange(nx + 4) - 2) % nx,
            (np.arange(ny + 4) - 2) % ny,
            nx,
            ny,
            scenario,
            conf.dtype,
        )
    else:
        shape = (px * (conf.nx + 4), py * (conf.ny + 4))
        sharding = NamedSharding(mesh, P("x", "y"))
        # Cache one local partition's field construction across all State leaves.
        cache: dict[tuple[int, int, int, int], dict[str, np.ndarray]] = {}

        def callback(name: str, index: tuple[slice, ...] | None) -> np.ndarray:
            if index is None:
                index = (slice(0, shape[0]), slice(0, shape[1]))
            sx, sy = index
            key = (
                sx.start or 0,
                sx.stop or shape[0],
                sy.start or 0,
                sy.stop or shape[1],
            )
            if key not in cache:
                ax, ay = np.arange(key[0], key[1]), np.arange(key[2], key[3])
                ix = (ax // (conf.nx + 4) * conf.nx + ax % (conf.nx + 4) - 2) % nx
                iy = (ay // (conf.ny + 4) * conf.ny + ay % (conf.ny + 4) - 2) % ny
                cache[key] = _fields(ix, iy, nx, ny, scenario, conf.dtype)
            return cache[key][name]

        values = {
            name: jax.make_array_from_callback(shape, sharding, partial(callback, name))
            for name in VARIABLES
        }
    return initialize_model(
        mesh=mesh,
        settings_overrides=options,
        physical_overrides=physical_overrides,
        state_overrides=values,
    )


def _step_local(
    vs: State, conf: Configuration, phys: PhysicalConstants
) -> tuple[State, Diagnostics]:
    """Compose unchanged physics kernels in the dynamics notebook's order."""
    from veris.advection import Advection
    from veris.area_mass import AreaWS, SeaIceMass
    from veris.clean_up import clean_up_advection, ridging
    from veris.dynamics_routines import SeaIceStrength
    from veris.dynsolver import IceVelocities, WindForcingXY
    from veris.fill_overlap import fill_overlap
    from veris.ocean_stress import OceanStressUV

    def assign(state: State, names: str, values: tuple[jax.Array, ...]) -> State:
        return replace(state, **dict(zip(names.split(), values, strict=True)))

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
    stress_u, stress_v = OceanStressUV(vs, conf, phys)
    vs = assign(vs, "hIceMean hSnowMean Area", Advection(vs, conf, phys))
    vs = assign(
        vs,
        "hIceMean hSnowMean Area TSurf os_hIceMean os_hSnowMean",
        clean_up_advection(vs, conf, phys),
    )
    vs = replace(vs, Area=ridging(vs, conf, phys))
    zeros = jnp.zeros_like(vs.Area)
    diagnostics = Diagnostics(zeros, stress_u, stress_v, zeros, zeros)
    return jax.tree.map(lambda array: fill_overlap(array, conf), (vs, diagnostics))


def step_with_diagnostics(
    vs: State, conf: Configuration, phys: PhysicalConstants
) -> tuple[State, Diagnostics]:
    """Advance dynamics and transport; return ocean stress outside State."""
    if conf.use_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        if set(mesh.axis_names) != {"x", "y"}:
            raise ValueError(
                "sharded stepping requires an active mesh with axes x and y"
            )
        if not {"x", "y"} <= set(mesh.manual_axes):
            return jax.shard_map(
                lambda state: _step_local(state, conf, phys),
                mesh=mesh,
                in_specs=P("x", "y"),
                out_specs=(P("x", "y"), P("x", "y")),
            )(vs)
    return _step_local(vs, conf, phys)


def step(vs: State, conf: Configuration, phys: PhysicalConstants) -> State:
    """Advance one 600-second reference step without thermodynamics."""
    return step_with_diagnostics(vs, conf, phys)[0]


compiled_step = jax.jit(step, static_argnames=("conf", "phys"))


@click.command(help=__doc__)
@click.option("--steps", type=click.IntRange(min=0), default=100)
@click.option("--backend", type=click.Choice(["cpu", "gpu"]), default="cpu")
@click.option("--nx", type=click.IntRange(min=2), default=1024)
@click.option("--ny", type=click.IntRange(min=2))
@click.option("--evp-steps", type=click.IntRange(min=1), default=120)
@output_options("dynamics.nc", tuple(VARIABLES))
def cli(**kwargs: Any) -> SimpleNamespace:
    """Parse the standalone experiment options."""
    return SimpleNamespace(**kwargs)


def main(argv: list[str] | None = None) -> None:
    """Integrate the reference experiment and save physical fields as netCDF."""
    args = parse_options(cli, argv)
    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices(args.backend)[0]):
        state, conf, phys = initialize(
            args.nx, args.ny, settings_overrides={"nEVPsteps": args.evp_steps}
        )
        with make_output(args, conf.deltatDyn) as manager:
            observe, select = output_callbacks(manager, conf.deltatDyn)
            state, compiled, elapsed = run_timed(
                state,
                partial(compiled_step, conf=conf, phys=phys),
                args.steps,
                observe=observe,
                select=select,
            )
        output = {
            name: np.asarray(getattr(state, name))[2:-2, 2:-2] for name in VARIABLES
        }
    if not all(np.isfinite(value).all() for value in output.values()):
        raise RuntimeError("nonfinite dynamics output")
    save_final(
        args,
        state,
        timedelta(seconds=args.steps * conf.deltatDyn),
        manager.settings,
        conf,
        phys,
    )
    print(
        f"dynamics: {args.steps} steps, {args.backend}, compile {compiled:.3f}s, run {elapsed:.3f}s, {args.output}"
    )


if __name__ == "__main__":
    main()
