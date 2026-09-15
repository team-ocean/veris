"""Restart equivalence for coupled dynamics, transport and thermodynamics.

Exercise real halo-free h5netcdf snapshots on a periodic rectangular island
case with spatially varying ice/wind and time-varying cooling. Reconstruct a
fresh serial State and its halos explicitly; no automatic restart API or
cross-device equivalence is assumed. Compare every State field exactly on the
same backend, including boundary halos, without relaxing numerical tolerances.
"""

from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris._typing import State
from veris.configuration import Configuration
from veris.fill_overlap import fill_overlap
from veris.initialization import initialize
from veris.io import read_record, update_state, write_snapshot
from veris.physical_constants import PhysicalConstants
from veris.setups import artificial
from veris.variables import VARIABLES


def assert_same_state(actual: State, expected: State) -> None:
    """Report concise per-field discrepancies, including nonfinite results."""
    for name in VARIABLES:
        left, right = (
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(expected, name)),
        )
        assert left.dtype == right.dtype, f"ERROR restart dtype changed: {name}"
        assert np.isfinite(left).all() and np.isfinite(right).all(), (
            f"ERROR restart nonfinite field: {name}"
        )
        assert np.array_equal(left, right), (
            f"ERROR restart mismatch: {name}, max_abs={np.max(np.abs(left - right)):.6g}"
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("checkpoint_step", [1, 3])
def test_coupled_restart_matches_uninterrupted_run(
    tmp_path: Path, dtype: str, checkpoint_step: int
) -> None:
    """A full physical snapshot plus reconstructed halos resumes the same trajectory."""
    initial, conf, phys = artificial.initialize(
        nx=6,
        ny=8,
        dtype=dtype,
        settings_overrides={"nEVPsteps": 3},
        physical_overrides={"rhoAir": 1.25},
    )
    x = jnp.arange(10, dtype=getattr(jnp, dtype))[:, None] - 2
    y = jnp.arange(12, dtype=getattr(jnp, dtype))[None, :] - 2
    pattern = jnp.sin(2 * jnp.pi * x / 6) * jnp.cos(2 * jnp.pi * y / 8)
    wind = 2 + pattern
    initial = replace(
        initial,
        hIceMean=initial.hIceMean * (1 + 0.15 * pattern),
        uWind=wind,
        wSpeed=jnp.abs(wind),
    )
    initial = jax.tree.map(lambda array: fill_overlap(array, conf), initial)
    cooling = (70.0, 110.0, 85.0, 130.0, 95.0)
    uninterrupted = initial
    checkpoint = initial
    for index, forcing in enumerate(cooling, start=1):
        uninterrupted = artificial.compiled_step(uninterrupted, conf, phys, forcing)
        if index == checkpoint_step:
            checkpoint = uninterrupted
    assert not np.array_equal(initial.hIceMean, uninterrupted.hIceMean)
    assert np.max(np.abs(np.asarray(uninterrupted.uIce))) > 0

    path = tmp_path / "restart.nc"
    write_snapshot(
        path,
        checkpoint,
        elapsed=timedelta(seconds=checkpoint_step * float(conf.deltatTherm)),
        conf=conf,
        phys=phys,
    )
    record = read_record(path)
    assert set(record.fields) == set(VARIABLES)
    assert all(array.shape == (6, 8) for array in record.fields.values())
    fresh, restored_conf, restored_phys = initialize(
        settings_overrides={
            f.name: record.configuration[f.name]
            for f in fields(Configuration)
            if f.init
        },
        physical_overrides={
            f.name: tuple(record.physical_constants[f.name])
            if isinstance(record.physical_constants[f.name], list)
            else record.physical_constants[f.name]
            for f in fields(PhysicalConstants)
            if f.init
        },
    )
    assert restored_conf == conf and restored_phys == phys
    # Poison every fresh field: no physical data or halo may be inherited from
    # the uninterrupted trajectory or accidentally supplied by initialization.
    fresh = jax.tree.map(lambda array: jnp.full_like(array, jnp.nan), fresh)
    restored = update_state(fresh, record)
    assert np.isnan(np.asarray(restored.hIceMean)[:2]).all()
    assert record.time == checkpoint_step * float(restored_conf.deltatTherm)
    resume_step = int(record.time / restored_conf.deltatTherm)
    assert resume_step == checkpoint_step

    if dtype == "float64" and checkpoint_step == 1:
        stale = artificial.compiled_step(
            restored, restored_conf, restored_phys, cooling[resume_step]
        )
        correct = artificial.compiled_step(checkpoint, conf, phys, cooling[resume_step])
        assert not np.array_equal(
            np.asarray(stale.hIceMean)[2:-2, 2:-2],
            np.asarray(correct.hIceMean)[2:-2, 2:-2],
        ), "ERROR restart oracle must detect omitted halo reconstruction"

    restored = jax.tree.map(lambda array: fill_overlap(array, restored_conf), restored)
    assert_same_state(restored, checkpoint)
    for forcing in cooling[resume_step:]:
        restored = artificial.compiled_step(
            restored, restored_conf, restored_phys, forcing
        )
    assert_same_state(restored, uninterrupted)
