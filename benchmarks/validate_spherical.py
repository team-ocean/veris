"""Validate long spherical transport, manufactured tensors and explicit status.

Run ``python -m benchmarks.validate_spherical --steps 1000 --output report.json``.
The float64 transport case is smooth periodic zonal rotation on a bounded
latitude band. Only the native extensive advection operator is advanced; no
thermodynamics, cleanup or ridging participates. Independent northward face
integrals and tensor refinement distinguish geometry from conservation. Status
probes are separate real thermodynamic transitions, not invented transport flags.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from veris import Experiment, Forcing, Grid, Simulation, advance, initialize_experiment
from veris.grid.geometry import OceanGeometry
from veris.grid.halos import fill_overlap, fill_state_overlap
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.physics.dynamics.rheology import strainrates, stressdiv
from veris.physics.transport.advection import calc_Advection


def _geometry(nx: int, latitude: np.ndarray) -> OceanGeometry:
    """Prescribe authoritative midpoint areas and native west/south faces."""
    radius = 6.370e6
    edges = np.r_[
        latitude[0] - (latitude[1] - latitude[0]) / 2,
        (latitude[:-1] + latitude[1:]) / 2,
        latitude[-1] + (latitude[-1] - latitude[-2]) / 2,
    ]
    dx = radius * 2 * np.pi / nx
    dy = radius * np.deg2rad(np.diff(edges))
    dyu = radius * np.deg2rad(np.r_[latitude[1] - latitude[0], np.diff(latitude)])
    shape = (nx, len(latitude))
    mask = np.ones((*shape, 1))
    south = mask.copy()
    south[:, 0] = 0
    area = np.broadcast_to(dx * np.cos(np.deg2rad(latitude)) * dy, shape)
    area_v = np.broadcast_to(dx * np.cos(np.deg2rad(edges[:-1])) * dyu, shape)
    return OceanGeometry(
        maskT=mask,
        maskU=mask,
        maskV=south,
        ht=np.full(shape, 1000.0),
        coriolis_t=np.zeros(shape),
        dxt=np.full(nx, dx),
        dxu=np.full(nx, dx),
        dyt=dy,
        dyu=dyu,
        area_t=area,
        area_u=area,
        area_v=area_v,
        latitude=latitude,
        latitude_edges=edges,
        radius=radius,
    )


def _radius(geometry: OceanGeometry) -> float:
    """Read the required spherical radius, rejecting incomplete geometry."""
    if geometry.radius is None:
        raise ValueError("spherical geometry requires radius")
    return float(geometry.radius)


def _transport_case(
    *, nx: int = 32, ny: int = 8, latitude: np.ndarray | None = None
) -> tuple[State, Configuration, PhysicalConstants, OceanGeometry]:
    """Initialize a physical spherical band with one precision policy."""
    lat = np.linspace(48.0, 76.0, ny) if latitude is None else latitude
    geometry = _geometry(nx, lat)
    state, conf, p = initialize_experiment(
        Experiment(
            grid=Grid(nx, len(lat)),
            geometry=geometry,
            settings={
                "dtype": "float64",
                "enable_cyclic_y": False,
                "noSlip": False,
                "deltatDyn": 3600.0,
                "deltatTherm": 3600.0,
            },
            constants={"radius": _radius(geometry)},
            initial_conditions={
                "hIceMean": 1.0,
                "hSnowMean": 0.0,
                "Area": 0.5,
                "TSurf": 260.0,
            },
            output=None,
        )
    )
    phi = jnp.deg2rad(jnp.pad(jnp.asarray(lat), 2, mode="edge"))
    longitude = 2 * jnp.pi * jnp.arange(nx) / nx
    field = 1 + 0.1 * jnp.sin(longitude[:, None]) * jnp.cos(
        jnp.deg2rad(jnp.asarray(lat))
    )
    state = replace(
        state,
        hIceMean=fill_overlap(jnp.pad(field, 2), conf),
        uIce=jnp.broadcast_to(
            1e-6 * _radius(geometry) * jnp.cos(phi), state.Area.shape
        ),
        vIce=jnp.zeros_like(state.vIce),
    )
    return fill_state_overlap(state, conf), conf, p, geometry


def _check_spherical_tendency(
    state: State, conf: Configuration, p: PhysicalConstants, geometry: OceanGeometry
) -> float:
    """Compare native uniform-north transport to an independent face integral."""
    speed = 0.1
    current = replace(
        state,
        uIce=jnp.zeros_like(state.uIce),
        vIce=fill_overlap(jnp.full_like(state.vIce, speed), conf, boundary="normal"),
    )
    after = calc_Advection(current, conf, jnp.ones_like(state.hIceMean))
    lat = np.asarray(geometry.latitude)
    edges = np.asarray(geometry.latitude_edges)
    radius = _radius(geometry)
    dx = radius * 2 * np.pi / conf.nx
    area = np.asarray(geometry.area_t)[0]
    exact = -conf.deltatTherm * speed * dx * np.diff(np.cos(np.deg2rad(edges))) / area
    actual = np.asarray(after[2:-2, 3:-3]) - 1
    reference = exact[1:-1][None, :]
    relative = float(np.max(np.abs(actual - reference)) / np.max(np.abs(reference)))
    if not np.isfinite(relative) or relative > 1e-11:
        raise ValueError(f"ERROR spherical tendency relative error={relative:.6g}")
    assert np.all(np.abs(lat) < 90)
    return relative


def _refinement() -> tuple[list[float], float]:
    """Check tensor divergence on shared latitudes and exact rigid rotation."""
    errors = []
    maximum_strain = 0.0
    for spacing in (4.0, 2.0, 1.0):
        lat = np.arange(40.0 + spacing / 2, 76.0, spacing)
        state, conf, p, geometry = _transport_case(nx=8, latitude=lat)
        radius = _radius(geometry)
        phi = np.deg2rad(np.pad(lat, 2, mode="edge"))
        u = jnp.broadcast_to(1e-7 * radius * np.cos(phi), state.Area.shape)
        strains = strainrates(state, conf, p, u, jnp.zeros_like(u))
        maximum_strain = max(
            maximum_strain, *(float(jnp.max(jnp.abs(e[2:-2, 4:-4]))) for e in strains)
        )
        s22 = jnp.broadcast_to(np.sin(phi), state.Area.shape)
        _, fy = stressdiv(state, conf, p, jnp.zeros_like(u), s22, jnp.zeros_like(u))
        face = np.deg2rad((lat[1:] + lat[:-1]) / 2)
        exact = np.cos(2 * face) / np.cos(face) / radius
        common = np.isclose(np.rad2deg(face)[:, None], np.arange(48.0, 69.0, 4.0)).any(
            axis=1
        )
        errors.append(
            float(np.max(np.abs(np.asarray(fy[3, 3:-2])[common] - exact[common])))
        )
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:]).tolist()
    if min(orders) < 1.8 or maximum_strain >= 1e-18:
        raise ValueError(
            f"ERROR spherical tensor refinement={orders}, rigid strain={maximum_strain:.6g}"
        )
    return orders, maximum_strain


def _status_evidence() -> dict[str, int | bool]:
    """Prove a real invalid intermediate step survives a later valid forcing."""
    model = Experiment.builtin(
        "growth",
        grid=Grid(2, 2),
        settings={"deltatTherm": 300.0, "deltatDyn": 300.0},
        constants={"saltIce_ref": 5.0},
        initial_conditions={"hIceMean": 0.0, "hSnowMean": 0.0, "Area": 0.0},
        forcing=Forcing(
            constant={"Qnet": 100.0, "Qsw": 0.0, "theta": 271.19},
            series={"ocSalt": np.array([0.0, 35.0])},
        ),
        steps=2,
        output=None,
    )
    state, conf, p = initialize_experiment(model)
    _, bad = advance(state, conf, p, model.forcing.at(0), physics="thermodynamics")
    _, good = advance(state, conf, p, model.forcing.at(1), physics="thermodynamics")
    first, control = int(jnp.max(bad.physics_status)), int(jnp.max(good.physics_status))
    rejected = False
    with Simulation.initialize(model) as simulation:
        try:
            simulation.run(2)
        except ValueError as error:
            if "invalid saline-ice formation" not in str(error):
                raise
            rejected = True
        accepted = simulation.completed_steps
    if (first, control, rejected, accepted) != (2, 0, True, 0):
        raise ValueError("ERROR intermediate native physics failure was lost")
    return {
        "first_invalid_status": first,
        "valid_control_status": control,
        "intermediate_failure_rejected": rejected,
        "accepted_steps_after_failure": accepted,
    }


def validate_spherical_transport(
    *, steps: int = 1000, nx: int = 32, ny: int = 8
) -> dict[str, Any]:
    """Return synchronized float64 transport/geometry/status acceptance evidence.

    Inventory and extrema are checked after every native transport iteration.
    The caller enables x64 before initialization. This modeled transport test
    does not certify climate skill or coupled-ocean stability.
    """
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if min(nx, ny) < 4:
        raise ValueError("nx and ny must be at least four")
    state, conf, p, geometry = _transport_case(nx=nx, ny=ny)
    tendency = _check_spherical_tendency(state, conf, p, geometry)
    area = 1 / state.recip_rA[2:-2, 2:-2]
    initial = state.hIceMean
    volume = jnp.sum(initial[2:-2, 2:-2] * area)
    initial_min, initial_max = (
        jnp.min(initial[2:-2, 2:-2]),
        jnp.max(initial[2:-2, 2:-2]),
    )

    @jax.jit
    def integrate(field: Array) -> tuple[Array, Array, Array, Array, Array]:
        def step(
            _: int, carry: tuple[Array, Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array, Array]:
            field, drift, low, high, finite = carry
            field = fill_overlap(calc_Advection(state, conf, field), conf)
            physical = field[2:-2, 2:-2]
            error = jnp.abs(jnp.sum(physical * area) / volume - 1)
            return (
                field,
                jnp.maximum(drift, error),
                jnp.minimum(low, jnp.min(physical)),
                jnp.maximum(high, jnp.max(physical)),
                finite & jnp.all(jnp.isfinite(field)),
            )

        return jax.lax.fori_loop(
            0,
            int(steps),
            step,
            (
                field,
                jnp.zeros_like(volume),
                initial_min,
                initial_max,
                jnp.asarray(True),
            ),
        )

    final, drift, minimum, maximum, finite = jax.block_until_ready(integrate(initial))
    if (
        not bool(finite)
        or float(drift) > 1e-10
        or float(minimum) < float(initial_min) - 1e-12
        or float(maximum) > float(initial_max) + 1e-12
    ):
        raise ValueError(
            f"ERROR long transport drift={float(drift):.6g}, bounds=[{float(minimum):.6g},{float(maximum):.6g}]"
        )
    lat = np.asarray(geometry.latitude)
    longitude = 2 * np.pi * np.arange(nx) / nx
    exact = 1 + 0.1 * np.sin(
        longitude[:, None] - 1e-6 * steps * conf.deltatTherm
    ) * np.cos(np.deg2rad(lat))
    rms = float(np.sqrt(np.mean((np.asarray(final[2:-2, 2:-2]) - exact) ** 2)))
    orders, strain = _refinement()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = (
        subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode
        != 0
    )
    return {
        "steps": int(steps),
        "dtype": str(final.dtype),
        "geometry": {
            "nx": nx,
            "ny": ny,
            "radius_m": _radius(geometry),
            "latitude_degrees": lat.tolist(),
            "area_convention": "supplied midpoint areas",
        },
        "maximum_relative_volume_drift": float(drift),
        "minimum": float(minimum),
        "maximum": float(maximum),
        "initial_minimum": float(initial_min),
        "initial_maximum": float(initial_max),
        "all_transport_steps_finite": bool(finite),
        "spherical_tendency_relative_error": tendency,
        "tensor_refinement_orders": orders,
        "rigid_rotation_maximum_strain": strain,
        "rms_rotation_error": rms,
        "status": _status_evidence(),
        "commit": revision.stdout.strip() or None,
        "dirty": dirty,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "platform": platform.platform(),
    }


def main() -> None:
    """Write a machine-readable report after all modeled acceptance checks pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    report = validate_spherical_transport(steps=args.steps, nx=args.nx, ny=args.ny)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS spherical transport:{args.steps} steps, volume drift={report['maximum_relative_volume_drift']:.3g}, tensor orders={report['tensor_refinement_orders']}"
    )


if __name__ == "__main__":
    main()
