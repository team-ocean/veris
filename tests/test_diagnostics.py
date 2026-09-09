"""Output-only coupling fields retain metadata without enlarging AD state."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import ModuleType

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_diagnostics_are_frozen_registered_pytree_with_complete_metadata(
    tmp_path: Path,
) -> None:
    """Metadata supplies valid defaults, dimensions and real netCDF attributes."""
    from veris.diagnostics import DIAGNOSTICS, Diagnostics
    from veris.variables import C_GRID, U_GRID, V_GRID, VARIABLES

    assert list(DIAGNOSTICS) == [field.name for field in fields(Diagnostics)]
    assert set(DIAGNOSTICS) == {
        "IcePenetSW",
        "OceanStressU",
        "OceanStressV",
        "EmPmR",
        "forc_salt_surface",
    }
    assert not DIAGNOSTICS.keys() & VARIABLES.keys()
    assert DIAGNOSTICS["OceanStressU"].dimensions == U_GRID
    assert DIAGNOSTICS["OceanStressV"].dimensions == V_GRID
    assert DIAGNOSTICS["EmPmR"].dimensions == C_GRID
    arrays = {
        name: jnp.full((6, 9), metadata.default, dtype=metadata.dtype)
        for name, metadata in DIAGNOSTICS.items()
    }
    diagnostics = Diagnostics(**arrays)
    leaves, structure = jax.tree.flatten(diagnostics)
    assert len(leaves) == 5
    rebuilt = jax.tree.unflatten(structure, leaves)
    assert isinstance(rebuilt, Diagnostics)
    with pytest.raises(FrozenInstanceError):
        setattr(diagnostics, "EmPmR", arrays["EmPmR"])  # noqa: B010 - test frozen guard
    path = tmp_path / "diagnostics.nc"
    with h5netcdf.File(path, "w") as output:
        output.dimensions = {"x_center": 6, "x_face": 6, "y_center": 9, "y_face": 9}
        for name, metadata in DIAGNOSTICS.items():
            variable = output.create_variable(name, metadata.dimensions, metadata.dtype)
            variable.attrs.update(metadata.netcdf_attributes())
            variable[:] = getattr(diagnostics, name)
    with h5netcdf.File(path, "r") as saved:
        for name, metadata in DIAGNOSTICS.items():
            actual = saved.variables[name]
            assert actual.dimensions == metadata.dimensions
            assert actual.attrs["units"] == metadata.units
            assert actual.attrs["long_name"] == metadata.long_name
            assert actual.attrs["description"] == metadata.description
            np.testing.assert_array_equal(actual[:], getattr(diagnostics, name))


def test_step_returns_separate_periodic_diagnostics_and_identical_state(
    halo: ModuleType,
) -> None:
    """Opting into coupling outputs preserves every calculated state field."""
    from veris.diagnostics import DIAGNOSTICS
    from veris.setup.artificial import initialize, step, step_with_diagnostics

    initial, sett, phys = initialize(5, 7)
    expected = step(initial, sett, phys, cooling=25.0)
    actual, diagnostics = step_with_diagnostics(initial, sett, phys, cooling=25.0)
    assert len(jax.tree.leaves(actual)) == 70
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    for name in DIAGNOSTICS:
        assert not hasattr(actual, name)
        field = np.asarray(getattr(diagnostics, name))
        assert field.shape == initial.iceMask.shape
        assert np.isfinite(field).all(), f"ERROR nonfinite diagnostic {name}"
        np.testing.assert_allclose(
            field, np.pad(field[2:-2, 2:-2], 2, mode="wrap"), atol=1e-13
        )
    assert np.any(np.asarray(diagnostics.OceanStressU) != 0)
