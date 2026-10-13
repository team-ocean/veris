"""Output-only coupling fields retain metadata without enlarging AD state."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import ModuleType

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris import Experiment, Grid, initialize_experiment
from veris.execution.advance import advance as advance_model


def test_diagnostics_are_frozen_registered_pytree_with_complete_metadata(
    tmp_path: Path,
) -> None:
    """Metadata supplies valid defaults, dimensions and real netCDF attributes."""
    from veris.model.diagnostics import DIAGNOSTICS, Diagnostics
    from veris.model.variables import C_GRID, U_GRID, V_GRID, VARIABLES

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
        name: jnp.full((6, 9), metadata.default, dtype="float64")
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
            variable = output.create_variable(name, metadata.dimensions, "float64")
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


def test_advance_returns_separate_periodic_diagnostics(
    halo: ModuleType,
) -> None:
    """Coupling fields remain separate from the 70 calculated State leaves."""
    from veris.model.diagnostics import DIAGNOSTICS

    initial, conf, phys = initialize_experiment(
        Experiment.builtin("island", grid=Grid(5, 7))
    )
    actual, diagnostics = advance_model(
        initial, conf, phys, {"Qnet": 25.0, "Qsw": 0.0}, physics="coupled"
    )
    assert len(jax.tree.leaves(actual)) == 70
    for name in DIAGNOSTICS:
        assert not hasattr(actual, name)
        field = np.asarray(getattr(diagnostics, name))
        assert field.shape == initial.iceMask.shape
        assert np.isfinite(field).all(), f"ERROR nonfinite diagnostic {name}"
        np.testing.assert_allclose(
            field, np.pad(field[2:-2, 2:-2], 2, mode="wrap"), atol=1e-13
        )
    assert np.any(np.asarray(diagnostics.OceanStressU) != 0)
