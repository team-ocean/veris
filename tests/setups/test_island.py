"""Standalone coupled integration with an island and no ocean model."""

from dataclasses import replace

import jax
import numpy as np
import pytest

from tests.island_helpers import compact_island
from veris import Experiment, initialize_experiment
from veris.execution.advance import advance


@pytest.mark.parametrize("profile", [None, "example"])
def test_island_defaults_preserve_land_area_basin_and_coast_masks(
    profile: str | None,
) -> None:
    """The sole example and unprofiled builtin share the resolved 800 m grid."""
    definition = Experiment.builtin("island", profile=profile)
    state, conf, _ = initialize_experiment(definition)
    expected = np.ones((80, 120))
    expected[30:50, 50:70] = 0
    mask = np.asarray(state.iceMask[2:-2, 2:-2])
    np.testing.assert_array_equal(mask, expected)
    assert definition.grid.nx * state.dxG[2, 2] == 64000
    assert definition.grid.ny * state.dyG[2, 2] == 96000
    assert np.sum(mask == 0) * 800**2 == 256000000
    assert conf.deltatDyn == conf.deltatTherm == 60
    assert conf.useAdaptiveEVP and conf.nEVPsteps == 400
    for name, axis in (("iceMaskU", 0), ("iceMaskV", 1)):
        np.testing.assert_array_equal(
            getattr(state, name)[2:-2, 2:-2], expected * np.roll(expected, 1, axis)
        )


def test_island_has_only_one_public_launch_profile() -> None:
    """The former refinement is the normal example, without a low-res launch."""
    from veris.setups.presets import get_preset

    assert tuple(get_preset("island").profiles) == ("example",)
    with pytest.raises(ValueError, match="unknown profile"):
        Experiment.builtin("island", profile="refined")


@pytest.mark.parametrize("half_width", [0, -1, True, 1.5])
def test_island_half_width_rejects_nonpositive_or_noninteger_counts(
    half_width: object,
) -> None:
    """Invalid geometry counts must fail through the host scenario validator."""
    with pytest.raises((TypeError, ValueError), match="islandHalfWidthCells"):
        Experiment.builtin("island", scenario={"islandHalfWidthCells": half_width})


def test_island_rejects_land_filling_a_global_dimension() -> None:
    """A resized island must leave ocean around itself on both axes."""
    with pytest.raises(ValueError, match="islandHalfWidthCells.*grid"):
        initialize_experiment(
            Experiment.builtin("island", scenario={"islandHalfWidthCells": 40})
        )


def test_island_masks_block_both_sides_of_coast() -> None:
    vs, _, _ = initialize_experiment(Experiment.builtin("island"))
    mask = np.asarray(vs.iceMask)
    assert mask.shape == (84, 124)
    assert np.any(mask == 0) and np.any(mask == 1)
    np.testing.assert_array_equal(vs.iceMaskU, mask * np.roll(mask, 1, axis=0))
    np.testing.assert_array_equal(vs.iceMaskV, mask * np.roll(mask, 1, axis=1))
    for name in ("hIceMean", "hSnowMean", "Area"):
        assert np.all(np.asarray(getattr(vs, name))[mask == 0] == 0)


def test_coupled_rest_equilibrium_is_preserved() -> None:
    from veris.model.physical_constants import PhysicalConstants

    constants = PhysicalConstants()

    vs, conf, phys = initialize_experiment(
        compact_island(
            scenario={
                "islandWindSpeed": 0,
                "islandAirTemperature": constants.celsius2K + constants.tempFrz,
            },
        )
    )
    initial = vs
    for _ in range(2):
        vs = advance(vs, conf, phys, {"Qnet": 0, "Qsw": 0.0})[0]
    for name in ("hIceMean", "hSnowMean", "Area", "uIce", "vIce"):
        np.testing.assert_allclose(
            getattr(vs, name), getattr(initial, name), atol=1e-10
        )


def test_coupled_forced_steps_keep_land_empty_and_halos_periodic() -> None:
    vs, conf, phys = initialize_experiment(compact_island())
    initial_ice = np.asarray(vs.hIceMean)
    for _ in range(3):
        vs = advance(vs, conf, phys, {"Qnet": 100, "Qsw": 0.0})[0]
    jax.block_until_ready(vs)
    for field in jax.tree.leaves(vs):
        assert np.all(np.isfinite(field)), "ERROR nonfinite integration field"
    for name in ("hIceMean", "hSnowMean", "Area", "uIce", "vIce"):
        array = np.asarray(getattr(vs, name))
        np.testing.assert_allclose(
            array, np.pad(array[2:-2, 2:-2], 2, mode="wrap"), atol=1e-13
        )
    for name, mask in (
        ("hIceMean", vs.iceMask),
        ("hSnowMean", vs.iceMask),
        ("Area", vs.iceMask),
        ("uIce", vs.iceMaskU),
        ("vIce", vs.iceMaskV),
    ):
        assert np.all(np.asarray(getattr(vs, name))[np.asarray(mask) == 0] == 0)
    assert np.all(np.asarray(vs.Area) >= 0) and np.all(np.asarray(vs.Area) <= 1)
    assert np.max(np.abs(np.asarray(vs.uIce))) > 0
    assert not np.allclose(vs.hIceMean, initial_ice)
    assert any(
        np.max(np.abs(np.asarray(value))) > 0
        for value in (vs.sigma1, vs.sigma2, vs.sigma12)
    )


def test_prescribed_heat_forcing_replaces_previous_outputs() -> None:
    """Prescribed cooling overwrites previous coupling fluxes on every call."""
    import jax.numpy as jnp

    vs, conf, phys = initialize_experiment(compact_island())
    changed = replace(
        vs, Qnet=jnp.full_like(vs.Qnet, -999), Qsw=jnp.full_like(vs.Qsw, -888)
    )
    expected = advance(vs, conf, phys, {"Qnet": 100, "Qsw": 0.0})[0]
    actual = advance(changed, conf, phys, {"Qnet": 100, "Qsw": 0.0})[0]
    for first, second in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(first, second)


def test_example_runs_in_fresh_process_without_mesh_helper() -> None:
    """Verify standalone import order without the serial pytest halo fixture."""
    import subprocess
    import sys

    code = """
from veris import Experiment, Grid, Execution, initialize_experiment

import jax
jax.config.update("jax_enable_x64", True)

from veris.execution.advance import advance
vs, conf, phys = initialize_experiment(Experiment.builtin('island', settings={'nEVPsteps': 2}))
result = advance(vs, conf, phys, {"Qnet": 100.0, "Qsw": 0.0})[0]
jax.block_until_ready(result)
assert result.hIceMean.shape == (84, 124)
assert bool((result.hIceMean >= 0).all())
print("standalone-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR standalone process failed: {result.stderr[-2000:]}"
    )
    assert result.stdout.strip() == "standalone-ok"
