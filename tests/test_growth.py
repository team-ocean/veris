"""Thermodynamic ice/snow budgets through the real Growth/solve4temp pipeline.

Construct radiative equilibrium to isolate latent-heat, freshwater, snow loading,
and mixed-layer exchange. Thicknesses are grid-cell means in metres; heat fluxes
are positive upward. Partial-cover cases catch accidental repeated area weights.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from conftest import StateFactory

from veris.configuration import Settings
from veris.growth import Growth
from veris.physical_constants import PhysicalConstants


def equilibrium_state(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, **changes: float
) -> Any:
    """Build saturated, isothermal atmosphere/ice/ocean with zero net forcing.

    The result inherits StateFactory's intentionally dynamic partial-state type.
    """
    temperature = phys.celsius2K + phys.tempFrz
    vapor_pressure = 10 ** (12.537 - 2663.5 / temperature)
    humidity = 0.622 * vapor_pressure / (100000 - 0.378 * vapor_pressure)
    fields = {
        "iceMask": 1,
        "hIceMean": 1,
        "hSnowMean": 0,
        "Area": 1,
        "TSurf": temperature,
        "LWdown": phys.stefBoltz * temperature**4,
        "SWdown": 0,
        "ATemp": temperature,
        "aqh": humidity,
        "wSpeed": 2,
        "fCori": 1e-4,
        "ocSalt": 34.7,
        "theta": temperature,
        "snowfall": 0,
        "precip": 0,
        "runoff": 0,
        "evap": 0,
        "Qnet": 0,
        "Qsw": 0,
        "os_hIceMean": 0,
        "os_hSnowMean": 0,
    }
    fields.update(changes)
    return state(
        **{key: np.broadcast_to(value, (3, 5)) for key, value in fields.items()}
    )


@pytest.mark.parametrize("area", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("categories", [1, 5])
def test_zero_forcing_preserves_equilibrium(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    area: float,
    categories: int,
) -> None:
    sett = replace(sett, nITC=categories)
    vs = equilibrium_state(state, sett, phys, Area=area, hIceMean=area)
    result = Growth(vs, sett, phys)
    expected = (
        area,
        0,
        area,
        phys.celsius2K + phys.tempFrz,
        0,
        0,
        0,
        0,
        phys.rhoIce * area,
        0,
        1 / np.sqrt(area**2 + sett.hIce_reg),
    )
    for actual, reference in zip(result, expected):
        assert actual.shape == (3, 5)
        np.testing.assert_allclose(actual, reference, atol=1e-10)


@pytest.mark.parametrize("cooling", [10.0, 100.0, 500.0])
@pytest.mark.parametrize("south", [False, True])
def test_open_water_freezing_conserves_latent_heat_and_freshwater(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    cooling: float,
    south: bool,
) -> None:
    sett = replace(sett, deltatTherm=600)
    phys = replace(phys, h0=0.5, h0_south=0.25)
    vs = equilibrium_state(
        state,
        sett,
        phys,
        hIceMean=0,
        Area=0,
        Qnet=cooling,
        fCori=-1e-4 if south else 1e-4,
    )
    ice, snow, area, _, freshwater, salt, _, residual, load, _, _ = Growth(
        vs, sett, phys
    )
    frozen_mass = cooling * sett.deltatTherm / phys.lhFusion
    expected_ice = frozen_mass / phys.rhoIce
    np.testing.assert_allclose(ice, expected_ice, rtol=1e-12)
    np.testing.assert_allclose(snow, 0, atol=1e-12)
    np.testing.assert_allclose(area, expected_ice * (4 if south else 2), rtol=1e-12)
    np.testing.assert_allclose(load, frozen_mass, rtol=1e-12)
    np.testing.assert_allclose(freshwater, cooling / phys.lhFusion, rtol=1e-12)
    np.testing.assert_allclose(salt, cooling / phys.lhFusion * 34.7 / phys.rhoFresh)
    np.testing.assert_allclose(residual, 0, atol=1e-10)


@pytest.mark.parametrize("heating", [0.0, 100.0, 500.0])
def test_ice_free_heating_and_shortwave_pass_to_ocean(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, heating: float
) -> None:
    vs = equilibrium_state(
        state, sett, phys, hIceMean=0, Area=0, Qnet=-heating, Qsw=-80
    )
    result = Growth(vs, sett, phys)
    for index in (0, 1, 2, 4, 5, 8, 9):
        np.testing.assert_allclose(result[index], 0, atol=1e-12)
    np.testing.assert_allclose(result[6], -80, atol=1e-12)
    np.testing.assert_allclose(result[7], -heating, atol=1e-12)


@pytest.mark.parametrize("area", [0.2, 0.7, 1.0])
@pytest.mark.parametrize("precip", [0.0, 2e-7])
def test_cold_snowfall_stores_water_on_ice_and_rain_reaches_leads(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    area: float,
    precip: float,
) -> None:
    snowfall = 1e-7
    vs = equilibrium_state(
        state, sett, phys, Area=area, snowfall=snowfall, precip=precip
    )
    result = Growth(vs, sett, phys)
    stored_mass = (snowfall + precip) * area * sett.deltatTherm * phys.rhoFresh
    np.testing.assert_allclose(result[0], 1, atol=1e-12)
    np.testing.assert_allclose(result[1] * phys.rhoSnow, stored_mass, atol=1e-10)
    np.testing.assert_allclose(result[8], phys.rhoIce + stored_mass, atol=1e-10)
    np.testing.assert_allclose(
        result[4], -precip * (1 - area) * phys.rhoFresh, atol=1e-12
    )


@pytest.mark.parametrize("snow", [0.1, 0.3, 0.5])
def test_flooding_converts_submerged_snow_and_preserves_column_mass(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, snow: float
) -> None:
    ice = 0.5
    vs = equilibrium_state(state, sett, phys, hIceMean=ice, hSnowMean=snow)
    result = Growth(vs, sett, phys)
    initial_mass = ice * phys.rhoIce + snow * phys.rhoSnow
    flooded_depth = max(0, initial_mass / phys.rhoSea - ice)
    np.testing.assert_allclose(result[0], ice + flooded_depth, atol=1e-12)
    np.testing.assert_allclose(
        result[1], snow - flooded_depth * phys.rhoIce / phys.rhoSnow, atol=1e-12
    )
    np.testing.assert_allclose(result[8], initial_mass, atol=1e-10)
    np.testing.assert_allclose(result[4], 0, atol=1e-12)


@pytest.mark.parametrize("warming", [-0.1, 0.0, 0.02])
def test_mixed_layer_melt_uses_ocean_heat_and_returns_freshwater(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, warming: float
) -> None:
    sett = replace(sett, deltatTherm=600)
    vs = equilibrium_state(
        state, sett, phys, theta=phys.celsius2K + phys.tempFrz + warming
    )
    result = Growth(vs, sett, phys)
    transfer = phys.stantonNr * phys.uStarBase * phys.rhoSea * phys.cpWater
    taper = 1 + (phys.McPheeTaperFac - 1) / (1 + np.exp((1 - 0.4) * 7 / 0.4))
    heat = transfer * max(0, warming) * taper
    melted_mass = heat * sett.deltatTherm / phys.lhFusion
    np.testing.assert_allclose(result[0], 1 - melted_mass / phys.rhoIce, atol=1e-12)
    np.testing.assert_allclose(result[4], -heat / phys.lhFusion, atol=1e-12)
    np.testing.assert_allclose(result[7], heat, atol=1e-10)
    assert np.all(np.asarray(result[2]) <= 1)
    assert np.all(np.asarray(result[2]) > 0)


@pytest.mark.parametrize("area", [0.25, 0.7])
def test_partial_cover_conduction_closes_latent_heat_budget(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, area: float
) -> None:
    # Remove concentration regularization to isolate pure energy accounting.
    sett = replace(sett, nITC=1, Area_reg=0)
    surface_temperature = 260.0
    freezing = phys.celsius2K + phys.tempFrz
    thickness = 1.5
    conduction = phys.iceConduct / thickness * (freezing - surface_temperature)
    vapor_pressure = 10 ** (12.537 - 2663.5 / surface_temperature)
    vs = equilibrium_state(
        state,
        sett,
        phys,
        Area=area,
        hIceMean=thickness * area,
        TSurf=surface_temperature,
        ATemp=surface_temperature,
        aqh=0.622 * vapor_pressure / (100000 - 0.378 * vapor_pressure),
        LWdown=phys.stefBoltz * surface_temperature**4 - conduction / phys.iceEmiss,
    )
    result = Growth(vs, sett, phys)
    latent_energy = (result[0] - vs.hIceMean) * phys.rhoIce * phys.lhFusion
    atmospheric_loss = area * conduction * sett.deltatTherm
    np.testing.assert_allclose(latent_energy, atmospheric_loss, rtol=1e-11)
    # All atmospheric heat loss made ice; none remains to cool the ocean.
    np.testing.assert_allclose(result[7], 0, atol=1e-10)


@pytest.mark.parametrize("area", [0.25, 0.7])
def test_partial_cover_shortwave_is_weighted_once(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, area: float
) -> None:
    sett = replace(sett, nITC=1)
    thickness = 1.5
    sunlight = 100.0
    vs = equilibrium_state(
        state,
        sett,
        phys,
        Area=area,
        hIceMean=thickness * np.sqrt(area**2 + sett.Area_reg),
        SWdown=sunlight,
        Qsw=-80,
    )
    result = Growth(vs, sett, phys)
    under_ice = (
        -sunlight * (1 - phys.dryIceAlb) * phys.shortwave * np.exp(-1.5 * thickness)
    )
    np.testing.assert_allclose(result[9], under_ice * area, rtol=1e-12)
    np.testing.assert_allclose(
        result[6], under_ice * area - 80 * (1 - area), rtol=1e-12
    )


@pytest.mark.parametrize("area", [0.25, 0.7])
def test_partial_cover_surface_heat_melts_snow_with_one_area_weight(
    state: StateFactory, sett: Settings, phys: PhysicalConstants, area: float
) -> None:
    sett = replace(sett, nITC=1, deltatTherm=600)
    phys = replace(phys, tempFrz=0)
    heating = 20.0
    vs = equilibrium_state(
        state,
        sett,
        phys,
        Area=area,
        hSnowMean=0.05,
        LWdown=phys.stefBoltz * phys.celsius2K**4 + heating / phys.snowEmiss,
    )
    result = Growth(vs, sett, phys)
    melted_mass = heating * area * sett.deltatTherm / phys.lhFusion
    np.testing.assert_allclose(result[0], 1, atol=1e-12)
    np.testing.assert_allclose(
        (vs.hSnowMean - result[1]) * phys.rhoSnow, melted_mass, rtol=1e-11
    )
    np.testing.assert_allclose(result[4], -melted_mass / sett.deltatTherm, atol=1e-12)


@pytest.mark.parametrize("area", [0.25, 0.7])
@pytest.mark.parametrize("heat_multiple", [1.0, 3.0])
def test_complete_snow_melt_and_excess_ice_melt_close_energy(
    state: StateFactory,
    sett: Settings,
    phys: PhysicalConstants,
    area: float,
    heat_multiple: float,
) -> None:
    """Surface heat first melts all snow; excess consumes ice latent heat."""
    sett = replace(sett, nITC=1, Area_reg=0, deltatTherm=600)
    phys = replace(phys, tempFrz=0)
    snow = 1e-4
    snow_latent = snow * phys.rhoSnow * phys.lhFusion
    heat = heat_multiple * snow_latent / (area * sett.deltatTherm)
    vs = equilibrium_state(
        state,
        sett,
        phys,
        Area=area,
        hSnowMean=snow,
        LWdown=phys.stefBoltz * phys.celsius2K**4 + heat / phys.snowEmiss,
    )
    result = Growth(vs, sett, phys)
    total_energy = heat * area * sett.deltatTherm
    expected_ice_loss = (total_energy - snow_latent) / (phys.rhoIce * phys.lhFusion)
    np.testing.assert_allclose(result[1], 0, atol=1e-13)
    np.testing.assert_allclose(result[0], 1 - expected_ice_loss, atol=1e-12)
    latent_used = (
        (vs.hIceMean - result[0]) * phys.rhoIce
        + (vs.hSnowMean - result[1]) * phys.rhoSnow
    ) * phys.lhFusion
    np.testing.assert_allclose(latent_used, total_energy, rtol=1e-10)
    np.testing.assert_allclose(result[7], 0, atol=1e-10)
