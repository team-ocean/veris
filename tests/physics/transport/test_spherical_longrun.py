"""Long native spherical transport needs both inventory and metric accuracy.

The control deliberately retains conservative face differencing with incorrect
Cartesian face lengths: conservation alone cannot certify spherical physics.
The status probe contains a real invalid native step followed by a valid one.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validate_spherical import (
    _check_spherical_tendency,
    _status_evidence,
    _transport_case,
    validate_spherical_transport,
)
from veris.grid.halos import fill_overlap
from veris.physics.transport.advection import calc_Advection


@pytest.mark.parametrize("steps", [8, 1000])
def test_long_transport_report_checks_inventory_and_geometry(steps: int) -> None:
    report = validate_spherical_transport(steps=steps, nx=16, ny=8)
    assert report["steps"] == steps
    assert report["dtype"] == "float64"
    assert report["geometry"]["radius_m"] == 6.370e6
    assert report["maximum_relative_volume_drift"] <= 1e-10
    assert report["minimum"] >= report["initial_minimum"] - 1e-12
    assert report["maximum"] <= report["initial_maximum"] + 1e-12
    assert report["spherical_tendency_relative_error"] <= 1e-11
    assert min(report["tensor_refinement_orders"]) >= 1.8
    assert report["rigid_rotation_maximum_strain"] < 1e-18
    assert np.isfinite(report["rms_rotation_error"])
    assert report["all_transport_steps_finite"] is True


def test_wrong_cartesian_metric_fails_spherical_tendency() -> None:
    state, conf, p, geometry = _transport_case(nx=16, ny=8)
    # Keep authoritative stored areas. Divergence still telescopes, even though
    # the northward face integral and resulting spherical tendency are wrong.
    assert geometry.radius is not None
    equatorial = float(geometry.radius) * 2 * np.pi / 16
    wrong = replace(
        state,
        dxG=jnp.full_like(state.dxG, equatorial),
        uIce=jnp.zeros_like(state.uIce),
        vIce=fill_overlap(jnp.full_like(state.vIce, 0.1), conf, boundary="normal"),
    )
    field = jnp.ones_like(state.hIceMean)
    after = calc_Advection(wrong, conf, field)
    area = np.asarray(1 / wrong.recip_rA[2:-2, 2:-2])
    relative = abs(np.sum((np.asarray(after[2:-2, 2:-2]) - 1) * area)) / np.sum(area)
    assert relative <= 1e-10
    with pytest.raises(ValueError, match="spherical tendency"):
        _check_spherical_tendency(wrong, conf, p, geometry)


def test_intermediate_physics_failure_is_retained() -> None:
    evidence = _status_evidence()
    assert evidence["first_invalid_status"] == 2
    assert evidence["valid_control_status"] == 0
    assert evidence["intermediate_failure_rejected"] is True
    assert evidence["accepted_steps_after_failure"] == 0


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_verifier_rejects_invalid_step_counts(steps: int) -> None:
    with pytest.raises((TypeError, ValueError), match="steps"):
        validate_spherical_transport(steps=steps)
