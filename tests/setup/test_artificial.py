"""Standalone coupled integration with an artificial island and no ocean model."""

import importlib
from types import ModuleType

import jax
import numpy as np
import pytest


@pytest.fixture
def example(halo: ModuleType) -> ModuleType:
    return importlib.import_module("veris.setup.artificial")


def test_artificial_masks_block_both_sides_of_coast(example: ModuleType) -> None:
    vs, _ = example.initialize()
    mask = np.asarray(vs.iceMask)
    assert mask.shape == (12, 16)
    assert np.any(mask == 0) and np.any(mask == 1)
    for i, j in np.ndindex(mask.shape):
        assert vs.iceMaskU[i, j] == mask[i, j] * mask[i - 1, j]
        assert vs.iceMaskV[i, j] == mask[i, j] * mask[i, j - 1]
    for name in ("hIceMean", "hSnowMean", "Area"):
        assert np.all(np.asarray(getattr(vs, name))[mask == 0] == 0)


def test_coupled_rest_equilibrium_is_preserved(example: ModuleType) -> None:
    from veris.settings import settings

    vs, sett = example.initialize(
        wind=0, air_temperature=settings["celsius2K"] + settings["tempFrz"]
    )
    initial = vs
    for _ in range(2):
        vs = example.step(vs, sett, cooling=0)
    for name in ("hIceMean", "hSnowMean", "Area", "uIce", "vIce"):
        np.testing.assert_allclose(
            getattr(vs, name), getattr(initial, name), atol=1e-10
        )


def test_coupled_forced_steps_keep_land_empty_and_halos_periodic(
    example: ModuleType,
) -> None:
    vs, sett = example.initialize()
    initial_ice = np.asarray(vs.hIceMean)
    for _ in range(3):
        vs = example.step(vs, sett, cooling=100)
    jax.block_until_ready(vs)
    for field in vs:
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


@pytest.mark.parametrize("nx, ny", [(1, 12), (8, 1)])
def test_too_small_grid_has_clear_error(example: ModuleType, nx: int, ny: int) -> None:
    with pytest.raises(ValueError, match="at least"):
        example.initialize(nx=nx, ny=ny)


def test_prescribed_forcing_replaces_previous_ocean_flux_outputs(
    example: ModuleType,
) -> None:
    """Ocean coupling outputs must not become next-step atmospheric forcing."""
    import jax.numpy as jnp

    vs, sett = example.initialize()
    changed = vs._replace(
        Qnet=jnp.full_like(vs.Qnet, -999), Qsw=jnp.full_like(vs.Qsw, -888)
    )
    expected = example.step(vs, sett, cooling=25)
    actual = example.step(changed, sett, cooling=25)
    for first, second in zip(actual, expected):
        np.testing.assert_array_equal(first, second)


def test_example_runs_in_fresh_process_without_mesh_helper() -> None:
    """Verify standalone import order without the serial pytest halo fixture."""
    import subprocess
    import sys

    code = """
import jax
jax.config.update("jax_enable_x64", True)
from veris.setup.artificial import initialize, step
vs, sett = initialize()
result = step(vs, sett)
jax.block_until_ready(result)
assert result.hIceMean.shape == (12, 16)
assert bool((result.hIceMean >= 0).all())
print("standalone-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR standalone process failed: {result.stderr[-2000:]}"
    )
    assert result.stdout.strip() == "standalone-ok"
