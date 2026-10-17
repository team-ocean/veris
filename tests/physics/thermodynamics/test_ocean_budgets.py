"""Independent zero-layer mass, salt and enthalpy budgets at the ocean boundary.

Mass fluxes are downward kg/m2/s, salt is kg salt/m2/s, and native Qnet is
upward W/m2. Stored ice/snow enthalpy relative to liquid at freezing is -L*m.
Analytic event tests suppress unrelated surface fluxes; status tests use the
real compiled advance. All balances use imposed boundary inputs, not a flux
reconstructed from the solver's final stock.
"""

from dataclasses import replace
from importlib import import_module
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import StateFactory

from veris import Experiment, Grid, initialize_experiment
from veris.execution.advance import advance
from veris.model.configuration import Configuration
from veris.model.physical_constants import PhysicalConstants
from veris.physics.thermodynamics.growth import Growth
from veris.physics.transport.cleanup import clean_up_advection


def column(state: StateFactory, p: PhysicalConstants, **changes: float) -> Any:
    """Build a homogeneous physical column with independently prescribed inputs."""
    t = float(p.celsius2K + p.tempFrz)
    fields = {
        "iceMask": 1.0,
        "hIceMean": 0.0,
        "hSnowMean": 0.0,
        "Area": 0.0,
        "TSurf": t,
        "LWdown": 0.0,
        "SWdown": 0.0,
        "ATemp": t,
        "aqh": 0.0,
        "wSpeed": 2.0,
        "fCori": 1e-4,
        "ocSalt": 35.0,
        "theta": t,
        "snowfall": 0.0,
        "precip": 0.0,
        "runoff": 0.0,
        "evap": 0.0,
        "Qnet": 0.0,
        "Qsw": 0.0,
        "os_hIceMean": 0.0,
        "os_hSnowMean": 0.0,
    }
    fields.update(changes)
    return state(**{k: np.full((3, 5), v) for k, v in fields.items()})


@pytest.fixture
def isolated_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exclude Newton/radiation/conduction from analytic phase-event tests."""

    def surface(
        vs: Any, conf: Configuration, p: PhysicalConstants, *args: Any
    ) -> tuple[jax.Array, ...]:
        z = jnp.zeros_like(vs.Area)
        return z + p.celsius2K + p.tempFrz, z, z, z, z

    monkeypatch.setattr(
        import_module("veris.physics.thermodynamics.growth"), "solve4temp", surface
    )


def balanced(
    result: tuple[jax.Array, ...],
    before: Any,
    p: PhysicalConstants,
    conf: Configuration,
    mass: float | np.ndarray,
    heat: float | np.ndarray,
    cleanup_ice: float = 0.0,
    cleanup_snow: float = 0.0,
) -> None:
    """Check independent boundary inputs against actual stock and ocean rates."""
    di = np.asarray(result[0] - before.hIceMean) - cleanup_ice
    ds = np.asarray(result[1] - before.hSnowMean) - cleanup_snow
    dm = p.rhoIce * di + p.rhoSnow * ds
    dt = conf.deltatTherm
    np.testing.assert_allclose(result[11] + dm / dt, mass, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(
        result[12] + p.rhoIce * di * p.saltIce_ref / (1000 * dt), 0, atol=1e-14
    )
    np.testing.assert_allclose(
        -result[7] - p.lhFusion * dm / dt, heat, rtol=1e-11, atol=1e-10
    )
    np.testing.assert_allclose(result[13], mass, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(result[14], heat, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(result[4], -result[11], atol=1e-14)
    np.testing.assert_allclose(
        result[5],
        (-result[11] * before.ocSalt + 1000 * result[12]) / p.rhoFresh,
        atol=1e-14,
    )


def test_pure_freezing_matches_actual_mass_and_salt_budget(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    isolated_surface: None,
) -> None:
    p = replace(phys, saltIce_ref=5.0)
    c = replace(conf, deltatTherm=300.0)
    before = column(state, p, Qnet=100.0)
    with jax.disable_jit():
        result = Growth(before, c, p)
    np.testing.assert_allclose(result[0], 100 * 300 / (900 * 334000), rtol=1e-11)
    np.testing.assert_allclose(result[5], 8.982035928143713e-6, rtol=1e-11)
    balanced(result, before, p, c, 0.0, -100.0)


@pytest.mark.parametrize("si", [0.0, 2.0, 5.0])
@pytest.mark.parametrize("q", [-100.0, 100.0])
def test_phase_change_nondefault_densities(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    isolated_surface: None,
    si: float,
    q: float,
) -> None:
    p = replace(phys, saltIce_ref=si, rhoIce=910.0, rhoSnow=350.0, rhoFresh=997.0)
    before = column(state, p, Qnet=q, Area=0.5, hIceMean=0.2)
    with jax.disable_jit():
        result = Growth(before, conf, p)
    balanced(result, before, p, conf, 0.0, -q * 0.5)
    assert np.sign(float(result[11][0, 0])) == -np.sign(q)


@pytest.mark.parametrize("area", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("event", ["snowfall", "precip", "evap", "runoff"])
def test_atmospheric_events_have_independent_mass_and_enthalpy(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    isolated_surface: None,
    area: float,
    event: str,
) -> None:
    p = replace(phys, saltIce_ref=0.0)
    rate = -1e-7 if event == "evap" else 1e-7
    before = column(state, p, Area=area, hIceMean=area * 0.25, **{event: rate})
    with jax.disable_jit():
        result = Growth(before, conf, p)
    mass = p.rhoFresh * rate * ((1 - area) if event == "evap" else 1)
    heat = -p.lhFusion * mass if event == "snowfall" else 0.0
    if event == "snowfall":
        expected_ice = area * 0.25 + mass * (1 - area) * conf.deltatTherm / p.rhoIce
        np.testing.assert_allclose(result[0], expected_ice, atol=1e-13)
    balanced(result, before, p, conf, mass, heat)
    if event == "snowfall":
        np.testing.assert_allclose(result[11], 0, atol=1e-14)
        np.testing.assert_allclose(result[7], 0, atol=1e-10)


def test_warm_open_water_snow_melts_with_latent_cooling(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    isolated_surface: None,
) -> None:
    before = column(state, phys, snowfall=1e-7, theta=280.0)
    with jax.disable_jit():
        result = Growth(before, conf, phys)
    np.testing.assert_allclose(result[0], 0, atol=1e-14)
    np.testing.assert_allclose(result[7], 33.4, rtol=1e-11)
    np.testing.assert_allclose(result[11], 1e-4, rtol=1e-11)
    balanced(result, before, phys, conf, 1e-4, -33.4)


@pytest.mark.parametrize("ice", [-1e-4, 5e-6])
def test_cleanup_exchange_compensates_actual_signed_stock_change(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    isolated_surface: None,
    ice: float,
) -> None:
    before = column(state, phys, Area=0.5, hIceMean=ice, hSnowMean=2e-6)
    with jax.disable_jit():
        cleaned = clean_up_advection(before, conf, phys)
        prepared = replace(
            before,
            hIceMean=cleaned[0],
            hSnowMean=cleaned[1],
            Area=cleaned[2],
            TSurf=cleaned[3],
            os_hIceMean=cleaned[4],
            os_hSnowMean=cleaned[5],
        )
        result = Growth(prepared, conf, phys)
    raw_minus_prepared_ice = float((before.hIceMean - prepared.hIceMean)[0, 0])
    raw_minus_prepared_snow = float((before.hSnowMean - prepared.hSnowMean)[0, 0])
    balanced(
        result,
        prepared,
        phys,
        conf,
        0.0,
        0.0,
        raw_minus_prepared_ice,
        raw_minus_prepared_snow,
    )


def test_flooding_converts_snow_without_spurious_latent_heat(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    isolated_surface: None,
) -> None:
    p = replace(phys, saltIce_ref=5.0)
    before = column(state, p, Area=1.0, hIceMean=0.01, hSnowMean=0.1)
    with jax.disable_jit():
        result = Growth(before, conf, p)
    assert float(result[0][0, 0]) > 0.01
    assert float(result[12][0, 0]) < 0
    np.testing.assert_allclose(result[11], 0, atol=1e-14)
    np.testing.assert_allclose(result[7], 0, atol=1e-10)
    balanced(result, before, p, conf, 0, 0)


@pytest.mark.parametrize("rate", [-1e-5, 1e-5, 1.0])
def test_actual_sublimation_or_deposition_is_a_boundary_event(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
    monkeypatch: pytest.MonkeyPatch,
    rate: float,
) -> None:
    def surface(
        vs: Any, c: Configuration, p: PhysicalConstants, *args: Any
    ) -> tuple[jax.Array, ...]:
        z = jnp.zeros_like(vs.Area)
        return z + p.celsius2K + p.tempFrz, z, z, z, z + rate

    monkeypatch.setattr(
        import_module("veris.physics.thermodynamics.growth"), "solve4temp", surface
    )
    before = column(state, phys, Area=0.5, hIceMean=0.02, hSnowMean=0.001)
    available = phys.rhoIce * 0.02 + phys.rhoSnow * 0.001
    actual = min(rate * 0.5, available / conf.deltatTherm)
    with jax.disable_jit():
        result = Growth(before, conf, phys)
    balanced(result, before, phys, conf, -actual, phys.lhFusion * actual)
    np.testing.assert_allclose(result[11], 0, atol=1e-14)


def test_dry_nan_operands_are_inactive(
    state: StateFactory,
    conf: Configuration,
    phys: PhysicalConstants,
) -> None:
    before = column(
        state,
        phys,
        iceMask=0.0,
        Qnet=float("nan"),
        ocSalt=float("nan"),
        snowfall=float("nan"),
        precip=float("nan"),
        evap=float("nan"),
        theta=float("nan"),
    )
    with jax.disable_jit():
        result = Growth(before, conf, phys)
    assert all(np.isfinite(np.asarray(x)).all() for x in result)
    for i in (0, 1, 2, 4, 5, 7, 11, 12, 13, 14):
        np.testing.assert_array_equal(result[i], 0)


@pytest.mark.parametrize(
    "q,si,so,status",
    [(100.0, 5.0, 0.0, 2.0), (100.0, 0.0, 0.0, 0.0), (-100.0, 5.0, 0.0, 0.0)],
)
def test_compiled_saline_formation_domain(
    q: float, si: float, so: float, status: float
) -> None:
    vs, c, p = initialize_experiment(
        Experiment.builtin(
            "growth",
            grid=Grid(2, 2),
            settings={"deltatTherm": 300.0, "deltatDyn": 300.0},
        )
    )
    z = jnp.zeros_like(vs.Area)
    vs = replace(
        vs,
        Area=z + (0.5 if q < 0 else 0.0),
        hIceMean=z + (0.2 if q < 0 else 0.0),
        hSnowMean=z,
        ocSalt=z + so,
        theta=z + 280.0,
        snowfall=z,
        precip=z,
        evap=z,
        runoff=z,
        ATemp=z + 280.0,
        LWdown=z + 1000.0,
        SWdown=z,
    )
    p = p.with_overrides(saltIce_ref=si)
    updated, diag = jax.jit(
        lambda s, p: advance(s, c, p, {"Qnet": q, "Qsw": 0.0}, physics="thermodynamics")
    )(vs, p)
    np.testing.assert_array_equal(diag.physics_status, status)
    if q < 0:
        assert float(diag.ocean_salt_mass_flux[2, 2]) > 0
    elif si > so:
        # Strong mixed-layer melting hides the simultaneous open-water formation.
        np.testing.assert_array_equal(updated.hIceMean, 0)


def test_helper_uses_thermodynamic_changes_and_signed_cleanup(
    conf: Configuration, phys: PhysicalConstants
) -> None:
    from veris.physics.thermodynamics.budgets import ocean_exchange_fluxes

    di = jnp.asarray([[1e-4, -2e-4]])
    ds = jnp.asarray([[2e-4, -1e-4]])
    ci = jnp.asarray([[-1e-5, 3e-5]])
    cs = jnp.asarray([[4e-5, -2e-5]])
    external = jnp.asarray([[1e-4, -2e-4]])
    p = phys.with_overrides(saltIce_ref=5.0)
    result = ocean_exchange_fluxes(
        ice_change=di,
        snow_change=ds,
        external_mass_input=external,
        cleanup_ice=ci,
        cleanup_snow=cs,
        configuration=conf,
        constants=p,
    )
    expected = (
        external - (p.rhoIce * (di - ci) + p.rhoSnow * (ds - cs)) / conf.deltatTherm
    )
    np.testing.assert_allclose(result.mass_input, expected, rtol=1e-12)
    np.testing.assert_allclose(
        result.salt_mass_input,
        -p.rhoIce * (di - ci) * 5 / (1000 * conf.deltatTherm),
        rtol=1e-12,
    )
    leaves, tree = jax.tree_util.tree_flatten(result)
    assert len(leaves) == 2
    assert type(jax.tree_util.tree_unflatten(tree, leaves)) is type(result)


def test_negative_bulk_ice_salinity_is_invalid() -> None:
    with pytest.raises(ValueError, match="saltIce_ref"):
        PhysicalConstants(saltIce_ref=-1.0)


def test_host_retains_intermediate_invalid_formation() -> None:
    import veris

    definition = Experiment.builtin(
        "growth",
        grid=Grid(2, 2),
        settings={"deltatTherm": 300.0, "deltatDyn": 300.0},
        constants={"saltIce_ref": 5.0},
        initial_conditions={
            "hIceMean": 0.0,
            "hSnowMean": 0.0,
            "Area": 0.0,
        },
        forcing=veris.Forcing(
            constant={"Qnet": 100.0, "Qsw": 0.0, "theta": 271.19},
            series={"ocSalt": np.array([0.0, 35.0])},
        ),
        steps=2,
    )
    with veris.Simulation.initialize(definition) as simulation:
        before = simulation.state
        with pytest.raises(ValueError, match="saline-ice formation"):
            simulation.run(2)
        assert simulation.state is before
        assert simulation.completed_steps == 0


def test_traced_negative_ice_salinity_returns_parameter_status() -> None:
    vs, c, p = initialize_experiment(Experiment.builtin("growth", grid=Grid(2, 2)))
    _, diag = jax.jit(
        lambda si: advance(
            vs, c, p.with_overrides(saltIce_ref=si), physics="thermodynamics"
        )
    )(jnp.asarray(-1.0))
    np.testing.assert_array_equal(diag.physics_status, 1)


def test_consumed_cleanup_is_not_applied_again_in_thermodynamic_continuation() -> None:
    vs, c, p = initialize_experiment(Experiment.builtin("growth", grid=Grid(2, 2)))
    z = jnp.zeros_like(vs.Area)
    vs = replace(
        vs, Area=z, hIceMean=z, hSnowMean=z, os_hIceMean=z + 1e-4, os_hSnowMean=z
    )
    forcing = {
        "Qnet": 0.0,
        "Qsw": 0.0,
        "theta": 280.0,
        "snowfall": 0.0,
        "precip": 0.0,
        "evap": 0.0,
        "runoff": 0.0,
    }
    first, d1 = advance(vs, c, p, forcing, physics="thermodynamics")
    _, d2 = advance(first, c, p, forcing, physics="thermodynamics")
    np.testing.assert_allclose(
        d1.ocean_mass_flux[2:-2, 2:-2], -p.rhoIce * 1e-4 / c.deltatTherm, rtol=1e-12
    )
    np.testing.assert_array_equal(first.os_hIceMean, 0)
    np.testing.assert_allclose(d2.ocean_mass_flux, 0, atol=1e-14)
