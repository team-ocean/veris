"""Variable metadata defines allocation and usable staggered netCDF output."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import h5netcdf
import numpy as np
import pytest


def test_calculation_registry_excludes_dead_fields_and_output_diagnostics() -> None:
    """Unused diagnostics must not enlarge the differentiated model state."""
    from veris.variables import VARIABLES

    assert len(VARIABLES) == 70
    assert {"hIceMean", "surfPress", "runoff", "rAz", "Fu"} <= VARIABLES.keys()
    assert (
        not {
            "saltflux",
            "OceanStressU",
            "OceanStressV",
            "IcePenetSW",
            "EmPmR",
            "forc_salt_surface",
            "dxC",
            "dyC",
            "rA",
            "rAu",
            "rAv",
            "recip_dxG",
            "recip_dyG",
            "recip_rAz",
        }
        & VARIABLES.keys()
    )


def test_variable_metadata_has_explicit_staggering_and_defaults() -> None:
    """Cartesian axes describe actual array storage, not geographic latitude."""
    from veris.variables import VARIABLES

    assert VARIABLES["theta"].dimensions == ("x_center", "y_center")
    assert VARIABLES["uIce"].dimensions == ("x_face", "y_center")
    assert VARIABLES["vIce"].dimensions == ("x_center", "y_face")
    assert VARIABLES["sigma12"].dimensions == ("x_face", "y_face")
    assert VARIABLES["theta"].units == "K"
    assert VARIABLES["hIceMean"].default == 0.0
    assert VARIABLES["iceMask"].default == 1.0
    for metadata in VARIABLES.values():
        assert metadata.long_name not in {"", "1"}
        assert metadata.description not in {"", "1"}
        assert metadata.units
        assert np.dtype(metadata.dtype) == np.dtype("float64")
    with pytest.raises(FrozenInstanceError):
        setattr(VARIABLES["theta"], "units", "degC")  # noqa: B010 - exercise frozen runtime guard


def test_wind_forcing_is_centered_before_stress_interpolation() -> None:
    """tauXY subtracts centered ice velocity from centered wind inputs."""
    from veris.variables import C_GRID, VARIABLES

    assert VARIABLES["uWind"].dimensions == C_GRID
    assert VARIABLES["vWind"].dimensions == C_GRID


def test_inverse_grid_metrics_have_physical_units_and_descriptions() -> None:
    """Inverse lengths and areas must not be exported as dimensionless data."""
    from veris.variables import VARIABLES

    for name in (
        "recip_dxC",
        "recip_dyC",
        "recip_dxU",
        "recip_dyU",
        "recip_dxV",
        "recip_dyV",
    ):
        assert VARIABLES[name].units == "m-1", name
        assert name.removeprefix("recip_") in VARIABLES[name].description
    for name in ("recip_rA", "recip_rAu", "recip_rAv"):
        assert VARIABLES[name].units == "m-2", name
        assert name.removeprefix("recip_") in VARIABLES[name].description


def test_metadata_drives_real_netcdf_roundtrip(tmp_path: Path) -> None:
    """Dimension names, dtypes and attributes can be passed directly to h5netcdf."""
    from veris.variables import VARIABLES

    path = tmp_path / "state.nc"
    dimensions = {"x_center": 6, "x_face": 6, "y_center": 9, "y_face": 9}
    with h5netcdf.File(path, "w") as output:
        output.dimensions = dimensions
        for name, metadata in VARIABLES.items():
            variable = output.create_variable(
                name, metadata.dimensions, dtype=metadata.dtype
            )
            variable.attrs.update(metadata.netcdf_attributes())
            variable[:] = np.full((6, 9), metadata.default)
    with h5netcdf.File(path, "r") as saved:
        assert set(saved.variables) == set(VARIABLES)
        for name, metadata in VARIABLES.items():
            variable = saved.variables[name]
            assert variable.dimensions == metadata.dimensions
            assert variable.attrs["units"] == metadata.units
            assert variable.attrs["long_name"] == metadata.long_name
            assert variable.attrs["description"] == metadata.description
            np.testing.assert_array_equal(variable[:], metadata.default)
