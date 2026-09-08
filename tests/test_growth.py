"""Thermodynamic ice/snow budgets through the real Growth/solve4temp pipeline.

Construct radiative equilibrium to isolate latent-heat, freshwater, snow loading,
and mixed-layer exchange. Thicknesses are grid-cell means in metres; heat fluxes
are positive upward. Partial-cover cases catch accidental repeated area weights.
"""

from typing import Any

import numpy as np
import pytest
from conftest import StateFactory

from veris.growth import Growth
from veris.state import Settings


def equilibrium_state(state: StateFactory, sett: Settings, **changes: float) -> Any:
    """Build saturated, isothermal atmosphere/ice/ocean with zero net forcing.

    The result inherits StateFactory's intentionally dynamic partial-state type.
    """
    temperature = sett.celsius2K + sett.tempFrz
    vapor_pressure = 10 ** (12.537 - 2663.5 / temperature)
    humidity = 0.622 * vapor_pressure / (100000 - 0.378 * vapor_pressure)
    fields = {
        "iceMask": 1,
        "hIceMean": 1,
        "hSnowMean": 0,
        "Area": 1,
        "TSurf": temperature,
        "LWdown": sett.stefBoltz * temperature**4,
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
    state: StateFactory, sett: Settings, area: float, categories: int
) -> None:
    sett = sett._replace(nITC=categories, recip_nITC=1 / categories)
    vs = equilibrium_state(state, sett, Area=area, hIceMean=area)
    result = Growth(vs, sett)
    expected = (
        area,
        0,
        area,
        sett.celsius2K + sett.tempFrz,
        0,
        0,
        0,
        0,
        sett.rhoIce * area,
        0,
        1 / np.sqrt(area**2 + sett.hIce_reg),
    )
    for actual, reference in zip(result, expected):
        assert actual.shape == (3, 5)
        np.testing.assert_allclose(actual, reference, atol=1e-10)


@pytest.mark.parametrize("cooling", [10.0, 100.0, 500.0])
@pytest.mark.parametrize("south", [False, True])
def test_open_water_freezing_conserves_latent_heat_and_freshwater(
    state: StateFactory, sett: Settings, cooling: float, south: bool
) -> None:
    sett = sett._replace(
        deltatTherm=600,
        recip_deltatTherm=1 / 600,
        recip_h0=2,
        recip_h0_south=4,
    )
    vs = equilibrium_state(
        state,
        sett,
        hIceMean=0,
        Area=0,
        Qnet=cooling,
        fCori=-1e-4 if south else 1e-4,
    )
    ice, snow, area, _, freshwater, salt, _, residual, load, _, _ = Growth(vs, sett)
    frozen_mass = cooling * sett.deltatTherm / sett.lhFusion
    expected_ice = frozen_mass / sett.rhoIce
    np.testing.assert_allclose(ice, expected_ice, rtol=1e-12)
    np.testing.assert_allclose(snow, 0, atol=1e-12)
    np.testing.assert_allclose(area, expected_ice * (4 if south else 2), rtol=1e-12)
    np.testing.assert_allclose(load, frozen_mass, rtol=1e-12)
    np.testing.assert_allclose(freshwater, cooling / sett.lhFusion, rtol=1e-12)
    np.testing.assert_allclose(salt, cooling / sett.lhFusion * 34.7 / sett.rhoFresh)
    np.testing.assert_allclose(residual, 0, atol=1e-10)


@pytest.mark.parametrize("heating", [0.0, 100.0, 500.0])
def test_ice_free_heating_and_shortwave_pass_to_ocean(
    state: StateFactory, sett: Settings, heating: float
) -> None:
    vs = equilibrium_state(state, sett, hIceMean=0, Area=0, Qnet=-heating, Qsw=-80)
    result = Growth(vs, sett)
    for index in (0, 1, 2, 4, 5, 8, 9):
        np.testing.assert_allclose(result[index], 0, atol=1e-12)
    np.testing.assert_allclose(result[6], -80, atol=1e-12)
    np.testing.assert_allclose(result[7], -heating, atol=1e-12)


@pytest.mark.parametrize("area", [0.2, 0.7, 1.0])
@pytest.mark.parametrize("precip", [0.0, 2e-7])
def test_cold_snowfall_stores_water_on_ice_and_rain_reaches_leads(
    state: StateFactory, sett: Settings, area: float, precip: float
) -> None:
    snowfall = 1e-7
    vs = equilibrium_state(state, sett, Area=area, snowfall=snowfall, precip=precip)
    result = Growth(vs, sett)
    stored_mass = (snowfall + precip) * area * sett.deltatTherm * sett.rhoFresh
    np.testing.assert_allclose(result[0], 1, atol=1e-12)
    np.testing.assert_allclose(result[1] * sett.rhoSnow, stored_mass, atol=1e-10)
    np.testing.assert_allclose(result[8], sett.rhoIce + stored_mass, atol=1e-10)
    np.testing.assert_allclose(
        result[4], -precip * (1 - area) * sett.rhoFresh, atol=1e-12
    )


@pytest.mark.parametrize("snow", [0.1, 0.3, 0.5])
def test_flooding_converts_submerged_snow_and_preserves_column_mass(
    state: StateFactory, sett: Settings, snow: float
) -> None:
    ice = 0.5
    vs = equilibrium_state(state, sett, hIceMean=ice, hSnowMean=snow)
    result = Growth(vs, sett)
    initial_mass = ice * sett.rhoIce + snow * sett.rhoSnow
    flooded_depth = max(0, initial_mass / sett.rhoSea - ice)
    np.testing.assert_allclose(result[0], ice + flooded_depth, atol=1e-12)
    np.testing.assert_allclose(
        result[1], snow - flooded_depth * sett.rhoIce / sett.rhoSnow, atol=1e-12
    )
    np.testing.assert_allclose(result[8], initial_mass, atol=1e-10)
    np.testing.assert_allclose(result[4], 0, atol=1e-12)


@pytest.mark.parametrize("warming", [-0.1, 0.0, 0.02])
def test_mixed_layer_melt_uses_ocean_heat_and_returns_freshwater(
    state: StateFactory, sett: Settings, warming: float
) -> None:
    sett = sett._replace(deltatTherm=600, recip_deltatTherm=1 / 600)
    vs = equilibrium_state(state, sett, theta=sett.celsius2K + sett.tempFrz + warming)
    result = Growth(vs, sett)
    transfer = sett.stantonNr * sett.uStarBase * sett.rhoSea * sett.cpWater
    taper = 1 + (sett.McPheeTaperFac - 1) / (1 + np.exp((1 - 0.4) * 7 / 0.4))
    heat = transfer * max(0, warming) * taper
    melted_mass = heat * sett.deltatTherm / sett.lhFusion
    np.testing.assert_allclose(result[0], 1 - melted_mass / sett.rhoIce, atol=1e-12)
    np.testing.assert_allclose(result[4], -heat / sett.lhFusion, atol=1e-12)
    np.testing.assert_allclose(result[7], heat, atol=1e-10)
    assert np.all(np.asarray(result[2]) <= 1)
    assert np.all(np.asarray(result[2]) > 0)


@pytest.mark.parametrize("area", [0.25, 0.7])
def test_partial_cover_conduction_closes_latent_heat_budget(
    state: StateFactory, sett: Settings, area: float
) -> None:
    # Remove concentration regularization to isolate pure energy accounting.
    sett = sett._replace(nITC=1, recip_nITC=1, Area_reg=0)
    surface_temperature = 260.0
    freezing = sett.celsius2K + sett.tempFrz
    thickness = 1.5
    conduction = sett.iceConduct / thickness * (freezing - surface_temperature)
    vapor_pressure = 10 ** (12.537 - 2663.5 / surface_temperature)
    vs = equilibrium_state(
        state,
        sett,
        Area=area,
        hIceMean=thickness * area,
        TSurf=surface_temperature,
        ATemp=surface_temperature,
        aqh=0.622 * vapor_pressure / (100000 - 0.378 * vapor_pressure),
        LWdown=sett.stefBoltz * surface_temperature**4 - conduction / sett.iceEmiss,
    )
    result = Growth(vs, sett)
    latent_energy = (result[0] - vs.hIceMean) * sett.rhoIce * sett.lhFusion
    atmospheric_loss = area * conduction * sett.deltatTherm
    np.testing.assert_allclose(latent_energy, atmospheric_loss, rtol=1e-11)
    # All atmospheric heat loss made ice; none remains to cool the ocean.
    np.testing.assert_allclose(result[7], 0, atol=1e-10)


@pytest.mark.parametrize("area", [0.25, 0.7])
def test_partial_cover_shortwave_is_weighted_once(
    state: StateFactory, sett: Settings, area: float
) -> None:
    sett = sett._replace(nITC=1, recip_nITC=1)
    thickness = 1.5
    sunlight = 100.0
    vs = equilibrium_state(
        state,
        sett,
        Area=area,
        hIceMean=thickness * np.sqrt(area**2 + sett.Area_reg),
        SWdown=sunlight,
        Qsw=-80,
    )
    result = Growth(vs, sett)
    under_ice = (
        -sunlight * (1 - sett.dryIceAlb) * sett.shortwave * np.exp(-1.5 * thickness)
    )
    np.testing.assert_allclose(result[9], under_ice * area, rtol=1e-12)
    np.testing.assert_allclose(
        result[6], under_ice * area - 80 * (1 - area), rtol=1e-12
    )


@pytest.mark.parametrize("area", [0.25, 0.7])
def test_partial_cover_surface_heat_melts_snow_with_one_area_weight(
    state: StateFactory, sett: Settings, area: float
) -> None:
    sett = sett._replace(
        nITC=1,
        recip_nITC=1,
        tempFrz=0,
        deltatTherm=600,
        recip_deltatTherm=1 / 600,
    )
    heating = 20.0
    vs = equilibrium_state(
        state,
        sett,
        Area=area,
        hSnowMean=0.05,
        LWdown=sett.stefBoltz * sett.celsius2K**4 + heating / sett.snowEmiss,
    )
    result = Growth(vs, sett)
    melted_mass = heating * area * sett.deltatTherm / sett.lhFusion
    np.testing.assert_allclose(result[0], 1, atol=1e-12)
    np.testing.assert_allclose(
        (vs.hSnowMean - result[1]) * sett.rhoSnow, melted_mass, rtol=1e-11
    )
    np.testing.assert_allclose(result[4], -melted_mass / sett.deltatTherm, atol=1e-12)


@pytest.mark.parametrize("area", [0.25, 0.7])
@pytest.mark.parametrize("heat_multiple", [1.0, 3.0])
def test_complete_snow_melt_and_excess_ice_melt_close_energy(
    state: StateFactory, sett: Settings, area: float, heat_multiple: float
) -> None:
    """Surface heat first melts all snow; excess consumes ice latent heat."""
    sett = sett._replace(
        nITC=1,
        recip_nITC=1,
        Area_reg=0,
        tempFrz=0,
        deltatTherm=600,
        recip_deltatTherm=1 / 600,
    )
    snow = 1e-4
    snow_latent = snow * sett.rhoSnow * sett.lhFusion
    heat = heat_multiple * snow_latent / (area * sett.deltatTherm)
    vs = equilibrium_state(
        state,
        sett,
        Area=area,
        hSnowMean=snow,
        LWdown=sett.stefBoltz * sett.celsius2K**4 + heat / sett.snowEmiss,
    )
    result = Growth(vs, sett)
    total_energy = heat * area * sett.deltatTherm
    expected_ice_loss = (total_energy - snow_latent) / (sett.rhoIce * sett.lhFusion)
    np.testing.assert_allclose(result[1], 0, atol=1e-13)
    np.testing.assert_allclose(result[0], 1 - expected_ice_loss, atol=1e-12)
    latent_used = (
        (vs.hIceMean - result[0]) * sett.rhoIce
        + (vs.hSnowMean - result[1]) * sett.rhoSnow
    ) * sett.lhFusion
    np.testing.assert_allclose(latent_used, total_energy, rtol=1e-10)
    np.testing.assert_allclose(result[7], 0, atol=1e-10)
