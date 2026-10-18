"""Independent spherical C-grid equations, work and definition contracts."""

from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from veris import Experiment, Grid, initialize_experiment
from veris.grid.geometry import OceanGeometry
from veris.grid.halos import fill_overlap
from veris.grid.ocean import initialize_from_ocean
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import State
from veris.physics.dynamics.rheology import strainrates, stressdiv
from veris.physics.transport.advection import calc_Advection


def spherical_geometry(
    latitude: Any = None, *, nx: int = 8, halos: bool = True
) -> OceanGeometry:
    """Hand-defined midpoint areas and distinct physical C-grid face positions."""
    lat = np.arange(48.0, 80.0, 4.0) if latitude is None else np.asarray(latitude)
    edges = np.r_[
        lat[0] - (lat[1] - lat[0]) / 2,
        (lat[:-1] + lat[1:]) / 2,
        lat[-1] + (lat[-1] - lat[-2]) / 2,
    ]
    radius = np.asarray(6.370e6)
    radius.setflags(write=False)
    dx = float(radius) * 2 * np.pi / nx
    dy = float(radius) * np.deg2rad(np.diff(edges))
    dy_face = float(radius) * np.deg2rad(np.r_[lat[1] - lat[0], np.diff(lat)])
    if halos:
        dxs = np.full(nx + 4, dx)
        dy = np.pad(dy, 2, mode="edge")
        dy_face = np.pad(dy_face, 2, mode="edge")
        lat_area = np.pad(lat, 2, mode="edge")
        edge_area = np.pad(edges[:-1], 2, mode="edge")
    else:
        dxs = np.full(nx, dx)
        lat_area, edge_area = lat, edges[:-1]
    shape = (len(dxs), len(dy))
    mask = np.ones((*shape, 1))
    area = np.broadcast_to(dx * np.cos(np.deg2rad(lat_area)) * dy, shape)
    area_v = np.broadcast_to(dx * np.cos(np.deg2rad(edge_area)) * dy_face, shape)
    return OceanGeometry(
        maskT=mask,
        maskU=mask,
        maskV=mask,
        ht=np.full(shape, 1000.0),
        coriolis_t=np.zeros(shape),
        dxt=dxs,
        dxu=dxs,
        dyt=dy,
        dyu=dy_face,
        area_t=area,
        area_u=area,
        area_v=area_v,
        latitude=lat,
        latitude_edges=edges,
        radius=radius,
    )


@pytest.fixture
def spherical_state() -> tuple[State, Configuration, PhysicalConstants]:
    """Eight by eight, four-degree latitude, closed walls, periodic longitude."""
    geometry = spherical_geometry()
    shape = np.shape(geometry.area_t)
    return initialize_from_ocean(
        geometry,
        settings_overrides={
            "deltatDyn": 3600.0,
            "deltatTherm": 3600.0,
            "enable_cyclic_y": False,
            "use_sharding": False,
            "noSlip": False,
        },
        state_overrides={"hIceMean": np.ones(shape), "Area": np.full(shape, 0.5)},
    )


def test_spherical_zonal_spacing_at_sixty_degrees(
    spherical_state: tuple[State, Configuration, PhysicalConstants],
) -> None:
    state, _, _ = spherical_state
    np.testing.assert_allclose(
        1 / state.recip_dxC[3, 5], 6.370e6 * np.deg2rad(45) * 0.5, rtol=1e-12
    )
    np.testing.assert_allclose(
        state.dxG[3, 5], 6.370e6 * np.deg2rad(45) * np.cos(np.deg2rad(58)), rtol=1e-12
    )


def test_uniform_northward_transport_matches_face_integral(
    spherical_state: tuple[State, Configuration, PhysicalConstants],
) -> None:
    state, conf, _ = spherical_state
    state = replace(state, vIce=jnp.full_like(state.vIce, 0.1))
    result = calc_Advection(state, conf, state.hIceMean)
    np.testing.assert_allclose(result[3, 5] - 1.0, 9.786682438863892e-5, rtol=1e-11)


def test_uniform_isotropic_stress_has_zero_force(
    spherical_state: tuple[State, Configuration, PhysicalConstants],
) -> None:
    state, conf, phys = spherical_state
    isotropic = jnp.full_like(state.Area, 1e4)
    fx, fy = stressdiv(
        state, conf, phys, isotropic, isotropic, jnp.zeros_like(isotropic)
    )
    np.testing.assert_allclose(fx[2:-2, 3:-3], 0.0, atol=1e-16)
    np.testing.assert_allclose(fy[2:-2, 3:-3], 0.0, atol=1e-16)


@pytest.mark.parametrize("southern", [False, True])
@pytest.mark.parametrize("no_slip", [False, True])
@pytest.mark.parametrize("coast_faces", [False, True])
def test_area_weighted_strain_stress_work_identity(
    southern: bool, no_slip: bool, coast_faces: bool
) -> None:
    lat = np.array([42.0, 45.0, 49.0, 54.0, 60.0, 65.0, 69.0, 72.0])
    if southern:
        lat = -lat[::-1]
    geometry = spherical_geometry(lat)
    if coast_faces:
        mask = np.array(geometry.maskT)
        mask[5, 6] = 0
        geometry = replace(
            geometry,
            maskT=mask,
            maskU=mask * np.roll(mask, 1, axis=0),
            maskV=mask * np.roll(mask, 1, axis=1),
        )
    state, conf, phys = initialize_from_ocean(
        geometry,
        settings_overrides={
            "enable_cyclic_y": False,
            "use_sharding": False,
            "noSlip": no_slip,
        },
    )
    rng = np.random.default_rng(512)
    fields = [
        fill_overlap(jnp.asarray(rng.normal(size=state.Area.shape)), conf)
        for _ in range(5)
    ]
    u, v, s11, s22, s12 = fields
    # Compact support removes boundary work; a dry island exercises masks.
    support = jnp.zeros_like(u).at[3:-3, 4:-4].set(1.0)
    u, v, s11, s22, s12 = [f * support for f in fields]
    if coast_faces:
        u, v = u * state.iceMaskU, v * state.iceMaskV
        s11, s22 = s11 * state.iceMask, s22 * state.iceMask
        assert np.all(np.asarray(state.iceMaskU[5:7, 6]) == 0)
        assert np.all(np.asarray(state.iceMaskV[5, 6:8]) == 0)
    wet = state.iceMask.at[5, 6].set(0.0)
    state = replace(state, iceMask=wet, maskInC=wet)
    e11, e22, e12 = strainrates(state, conf, phys, u, v)
    fx, fy = stressdiv(state, conf, phys, s11, s22, s12)
    strain_work = jnp.sum(
        (s11 * e11 + s22 * e22) / state.recip_rA + 2 * s12 * e12 * state.rAz
    )
    force_work = jnp.sum(u * fx / state.recip_rAu + v * fy / state.recip_rAv)
    scale = jnp.sum(
        jnp.abs(u * fx / state.recip_rAu) + jnp.abs(v * fy / state.recip_rAv)
    )
    assert abs(float((strain_work + force_work) / scale)) <= 1e-11


def test_solid_body_rotation_refines_at_second_order() -> None:
    errors = []
    for spacing in (4.0, 2.0, 1.0):
        lat = np.arange(40.0 + spacing / 2, 76.0, spacing)
        state, conf, phys = initialize_from_ocean(
            spherical_geometry(lat),
            settings_overrides={
                "enable_cyclic_y": False,
                "use_sharding": False,
                "noSlip": False,
            },
        )
        phi = np.deg2rad(np.pad(lat, 2, mode="edge"))
        u = jnp.broadcast_to(1e-7 * 6.370e6 * np.cos(phi), state.Area.shape)
        e11, e22, e12 = strainrates(state, conf, phys, u, jnp.zeros_like(u))
        assert float(jnp.max(jnp.abs(e11[2:-2, 4:-4]))) < 1e-18
        assert float(jnp.max(jnp.abs(e22[2:-2, 4:-4]))) < 1e-18
        assert float(jnp.max(jnp.abs(e12[2:-2, 4:-4]))) < 1e-18
        # A nonconstant axisymmetric tensor has north force d(s22)/dy - tan(phi)(s22-s11)/R.
        s11 = jnp.zeros_like(u)
        s22 = jnp.broadcast_to(np.sin(phi), u.shape)
        fx, fy = stressdiv(state, conf, phys, s11, s22, s11)
        face = np.deg2rad((lat[1:] + lat[:-1]) / 2)
        exact = (np.cos(2 * face) / np.cos(face)) / 6.370e6
        # Compare the same physical latitudes at every resolution.
        common = np.isclose(np.rad2deg(face)[:, None], np.arange(48.0, 69.0, 4.0)).any(
            axis=1
        )
        errors.append(
            float(np.max(np.abs(np.asarray(fy[3, 3:-2])[common] - exact[common])))
        )
        np.testing.assert_allclose(fx[2:-2, 4:-4], 0.0, atol=1e-18)
    assert min(np.log2(np.asarray(errors[:-1]) / errors[1:])) >= 1.8


@pytest.mark.parametrize(
    "overrides",
    [
        {"latitude": None},
        {"latitude_edges": None},
        {"radius": None},
        {"radius": 0.0},
        {"radius": np.inf},
        {"latitude": np.ones(7)},
        {"latitude": np.full(8, np.nan)},
        {"latitude": np.arange(8)[::-1]},
        {"latitude_edges": np.arange(9) + 90.0},
    ],
)
def test_invalid_spherical_description_fails_before_allocation(
    overrides: dict[str, Any],
) -> None:
    geometry = spherical_geometry()
    with pytest.raises(ValueError):
        initialize_from_ocean(replace(geometry, **overrides))


def test_spherical_definition_round_trip(tmp_path: Any) -> None:
    import json

    experiment = Experiment(
        grid=Grid(8, 8),
        geometry=spherical_geometry(halos=False),
        settings={"enable_cyclic_y": False},
    )
    path = tmp_path / "spherical.json"
    path.write_text(json.dumps(experiment.to_dict()))
    restored = Experiment.from_file(path)
    assert restored.geometry is not None
    np.testing.assert_array_equal(
        restored.geometry.latitude, np.arange(48.0, 80.0, 4.0)
    )
    actual, _, _ = initialize_experiment(restored)
    np.testing.assert_allclose(
        1 / actual.recip_dxC[3, 5], 6.370e6 * np.deg2rad(45) * 0.5, rtol=1e-12
    )


def test_spherical_coordinates_survive_non_square_partitioning(tmp_path: Any) -> None:
    """Real two-device x/y partitions retain authoritative nonuniform metrics."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    probe = """
import runpy
from dataclasses import replace
import jax
import numpy as np
from veris import Experiment, Execution, Grid, initialize_experiment
from veris.execution.runtime import create_mesh
from veris.grid.halos import remove_halos, fill_state_overlap
from veris.physics.dynamics.rheology import strainrates, stressdiv
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
geometry = runpy.run_path('tests/grid/test_spherical_operators.py')['spherical_geometry'](
    np.array([-72., -68., -63., -57., -50., -44., -39., -35.]), nx=12, halos=False)
definition = Experiment(grid=Grid(12, 8), geometry=geometry, settings={'enable_cyclic_y': False})
serial, serial_conf, phys = initialize_experiment(definition)
def prepare(state, conf):
    return fill_state_overlap(replace(state,
        uIce=.1*state.dxU/(6.370e6*2*np.pi/12)*state.iceMaskU,
        vIce=.02*jnp.sin(-state.k2AtC*6.370e6)*state.iceMaskV,
        sigma1=state.dxU/1000, sigma2=state.dyV/1000,
        sigma12=.01*state.k2AtC*6.370e6), conf)
def operators(state, conf):
    return (*strainrates(state, conf, phys, state.uIce, state.vIce),
            *stressdiv(state, conf, phys, state.sigma1, state.sigma2, state.sigma12))
serial = prepare(serial, serial_conf)
reference = operators(serial, serial_conf)
names = ('dxG', 'dyG', 'dxU', 'dyU', 'dxV', 'dyV', 'recip_dxC', 'recip_dyC',
         'recip_dxU', 'recip_dyU', 'recip_dxV', 'recip_dyV', 'k2AtC', 'k2AtZ',
         'rAz', 'recip_rA', 'recip_rAu', 'recip_rAv')
for partitions in ((2, 1), (1, 2)):
    definition = replace(definition, execution=Execution(partitions=partitions, backend='cpu'))
    mesh = create_mesh(definition.execution.partitions, 'cpu')
    with jax.set_mesh(mesh):
        parallel, parallel_conf, _ = initialize_experiment(definition, mesh=mesh)
        parallel = prepare(parallel, parallel_conf)
        for name in names:
            np.testing.assert_allclose(remove_halos(getattr(parallel, name), mesh),
                getattr(serial, name)[2:-2, 2:-2], rtol=1e-12, atol=1e-20,
                err_msg='ERROR spherical partition ' + name)
        results = jax.shard_map(lambda local: operators(local, parallel_conf),
            mesh=mesh, in_specs=P('x','y'), out_specs=P('x','y'))(parallel)
        for actual, expected in zip(results, reference, strict=True):
            np.testing.assert_allclose(remove_halos(actual, mesh), expected[2:-2, 2:-2],
                rtol=1e-12, atol=1e-18, err_msg='ERROR spherical partition operator')
print('spherical partition checks passed')
"""
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_NUM_CPU_DEVICES="2",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    log = tmp_path / "spherical-partition.log"
    with log.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=Path.cwd(),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    assert result.returncode == 0, (
        f"ERROR spherical partitions: {log.read_text()[-2500:]}"
    )


def test_spherical_closed_wall_shear_has_physical_line_traction(
    spherical_state: tuple[State, Configuration, PhysicalConstants],
) -> None:
    """A wall's half dual cell must give traction times its actual line length."""
    from veris.grid.halos import fill_state_overlap

    state, conf, phys = spherical_state
    conf = replace(conf, noSlip=True)
    state = fill_state_overlap(state, conf)
    zeros = jnp.zeros_like(state.Area)
    shear = zeros.at[:, 2].set(2.0).at[:, -2].set(-2.0)
    fx, _ = stressdiv(state, conf, phys, zeros, zeros, shear)
    np.testing.assert_allclose(
        fx[2:-2, 2], -2 * state.dxV[2:-2, 2] * state.recip_rAu[2:-2, 2], rtol=1e-12
    )
    np.testing.assert_allclose(
        fx[2:-2, -3], -2 * state.dxV[2:-2, -2] * state.recip_rAu[2:-2, -3], rtol=1e-12
    )


def test_spherical_adapter_defaults_to_closed_latitude_boundaries() -> None:
    """A newly authored spherical grid must not silently wrap north to south."""
    geometry = spherical_geometry()
    state, conf, _ = initialize_from_ocean(
        geometry, settings_overrides={"use_sharding": False}
    )
    assert conf.enable_cyclic_y is False
    np.testing.assert_array_equal(state.iceMask[:, :2], 0.0)
    np.testing.assert_array_equal(state.iceMask[:, -2:], 0.0)
    definition = Experiment(grid=Grid(8, 8), geometry=spherical_geometry(halos=False))
    assert definition.configuration.enable_cyclic_y is False


def test_spherical_geometry_rejects_cyclic_latitude() -> None:
    """Strictly monotone bounded latitude coordinates cannot form a periodic ring."""
    geometry = spherical_geometry()
    with pytest.raises(ValueError, match="cyclic|latitude"):
        initialize_from_ocean(geometry, settings_overrides={"enable_cyclic_y": True})
    definition_geometry = spherical_geometry(halos=False)
    with pytest.raises(ValueError, match="cyclic|latitude"):
        Experiment(
            grid=Grid(8, 8),
            geometry=definition_geometry,
            settings={"enable_cyclic_y": True},
        )


def test_spherical_scalar_gradients_use_distinct_face_positions(
    spherical_state: tuple[State, Configuration, PhysicalConstants],
) -> None:
    """Independent sine differences test the two normal gradient components."""
    state, conf, phys = spherical_state
    longitude_west = np.deg2rad(45 * np.arange(-2, 10))
    latitude_south = np.deg2rad(np.r_[46.0, 46.0, np.arange(46.0, 79.0, 4.0), 78.0])
    u = jnp.broadcast_to(jnp.asarray(np.sin(longitude_west))[:, None], state.Area.shape)
    v = jnp.broadcast_to(jnp.asarray(np.sin(latitude_south)), state.Area.shape)
    zeros = jnp.zeros_like(u)
    e11, _, _ = strainrates(state, conf, phys, u, zeros)
    _, e22, _ = strainrates(state, conf, phys, zeros, v)
    expected_x = (np.sin(longitude_west[3:11]) - np.sin(longitude_west[2:10]))[
        :, None
    ] / (6.370e6 * np.deg2rad(45) * np.cos(np.deg2rad(np.arange(48.0, 80.0, 4.0))))
    expected_y = (np.sin(latitude_south[3:11]) - np.sin(latitude_south[2:10])) / (
        6.370e6 * np.deg2rad(4)
    )
    np.testing.assert_allclose(e11[2:-2, 2:-2], expected_x, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(
        e22[2:-2, 2:-2], np.broadcast_to(expected_y, (8, 8)), rtol=1e-12
    )
