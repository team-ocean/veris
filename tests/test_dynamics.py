"""Shared dynamics stage preserves reference ordering and leaves halos to callers."""

from dataclasses import replace
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.advection import Advection
from veris.area_mass import AreaWS, SeaIceMass
from veris.clean_up import clean_up_advection, ridging
from veris.dynamics_routines import SeaIceStrength
from veris.dynsolver import IceVelocities, WindForcingXY
from veris.ocean_stress import OceanStressUV
from veris.setups.artificial import initialize


@pytest.mark.parametrize("cyclic_y", [False, True])
def test_dynamics_transport_preserves_intermediate_state(cyclic_y: bool) -> None:
    """An extra halo refresh or reordered stress/transport must change this result."""
    advance = import_module("veris.dynamics").dynamics_transport
    initial, conf, phys = initialize(
        5, 7, settings_overrides={"nEVPsteps": 2, "enable_cyclic_y": cyclic_y}
    )
    # Deliberately unrefreshed forcing halos detect a misplaced final refresh.
    initial = replace(
        initial,
        Qnet=jnp.arange(initial.Qnet.size, dtype=initial.Qnet.dtype).reshape(
            initial.Qnet.shape
        ),
    )
    expected = initial
    for names, kernel in (
        ("SeaIceMassC SeaIceMassU SeaIceMassV", SeaIceMass),
        ("AreaW AreaS", AreaWS),
        ("WindForcingX WindForcingY", WindForcingXY),
    ):
        expected = replace(
            expected,
            **dict(zip(names.split(), kernel(expected, conf, phys), strict=True)),
        )
    expected = replace(expected, SeaIceStrength=SeaIceStrength(expected, conf, phys))
    velocity = IceVelocities(expected, conf, phys)
    expected = replace(
        expected,
        uIce=velocity[0],
        vIce=velocity[1],
        sigma1=velocity[2],
        sigma2=velocity[3],
        sigma12=velocity[4],
    )
    stress = OceanStressUV(expected, conf, phys)
    ice, snow, area = Advection(expected, conf, phys)
    expected = replace(expected, hIceMean=ice, hSnowMean=snow, Area=area)
    ice, snow, area, temperature, ice_overshoot, snow_overshoot = clean_up_advection(
        expected, conf, phys
    )
    expected = replace(
        expected,
        hIceMean=ice,
        hSnowMean=snow,
        Area=area,
        TSurf=temperature,
        os_hIceMean=ice_overshoot,
        os_hSnowMean=snow_overshoot,
    )
    expected = replace(expected, Area=ridging(expected, conf, phys))
    actual = advance(initial, conf, phys)
    for result, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves((expected, *stress)), strict=True
    ):
        np.testing.assert_array_equal(result, reference)
