"""Closed C-grid walls retain their physical corner shear-stress values."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from veris.dynamics_routines import stressdiv
from veris.fill_overlap import fill_state_overlap
from veris.initialization import initialize
from veris.setups import artificial, run_dyn


@pytest.mark.parametrize("no_slip", [False, True])
def test_refresh_distinguishes_wall_shear_from_interior(no_slip: bool) -> None:
    """North wall stress lives at -2 and must never copy the center row -3."""
    state, conf, _ = initialize(
        5,
        6,
        settings_overrides={
            "use_sharding": False,
            "enable_cyclic_y": False,
            "noSlip": no_slip,
        },
    )
    shear = jnp.full_like(state.sigma12, 3.0)
    shear = shear.at[:, 2].set(7.0).at[:, -2].set(-11.0)
    refreshed = fill_state_overlap(replace(state, sigma12=shear), conf)
    np.testing.assert_array_equal(refreshed.sigma12[:, 2], 7.0 if no_slip else 0.0)
    np.testing.assert_array_equal(refreshed.sigma12[:, -2], -11.0 if no_slip else 0.0)
    np.testing.assert_array_equal(refreshed.sigma12[:, 3:-2], 3.0)
    np.testing.assert_array_equal(refreshed.sigma12[:, :2], 0.0)
    np.testing.assert_array_equal(refreshed.sigma12[:, -1], 0.0)
    np.testing.assert_array_equal(
        fill_state_overlap(refreshed, conf).sigma12, refreshed.sigma12
    )


def test_opposite_wall_shear_produces_equal_tangential_drag() -> None:
    """Equal signed drag requires opposite physical shear at south/north walls."""
    state, conf, phys = initialize(
        5,
        6,
        settings_overrides={
            "use_sharding": False,
            "enable_cyclic_y": False,
            "noSlip": True,
        },
    )
    zeros = jnp.zeros_like(state.sigma12)
    shear = zeros.at[:, 2].set(2.0).at[:, -2].set(-2.0)
    state = fill_state_overlap(
        replace(
            state,
            sigma12=shear,
            dxV=jnp.ones_like(shear),
            recip_rAu=jnp.ones_like(shear),
        ),
        conf,
    )
    force_u, _ = stressdiv(state, conf, phys, zeros, zeros, state.sigma12)
    np.testing.assert_array_equal(force_u[2:-2, 2], -2.0)
    np.testing.assert_array_equal(force_u[2:-2, -3], -2.0)
    np.testing.assert_array_equal(force_u[2:-2, 3:-3], 0.0)


def test_free_slip_momentum_steps_keep_both_wall_shears_zero() -> None:
    """Relaxation must not inherit spurious north traction from halo refresh."""
    state, conf, phys = artificial.initialize(
        6,
        8,
        settings_overrides={
            "enable_cyclic_y": False,
            "noSlip": False,
            "nEVPsteps": 2,
        },
    )
    # A nonzero interior stress makes accidental nearest-row copying visible
    # even with symmetric forcing and vanishing physical boundary shear.
    shear = jnp.zeros_like(state.sigma12).at[:, 3:-2].set(0.2)
    state = replace(state, sigma12=shear)
    for _ in range(2):
        state = run_dyn.step(state, conf, phys)
        np.testing.assert_array_equal(state.sigma12[:, 2], 0.0)
        np.testing.assert_array_equal(state.sigma12[:, -2], 0.0)
