"""Typed restart definition arrays and strict reference manifest validation."""

from pathlib import Path

import h5netcdf
import h5py
import numpy as np
import pytest

from veris.restart_arrays import ArrayManifest, hydrate_definition


def test_manifest_roundtrip_retains_bool_dtype_and_native_numeric_values(
    tmp_path: Path,
) -> None:
    """Typed variables avoid JSON grid copies and restore logical masks."""
    manifest = ArrayManifest()
    mask = np.array([[True, False]], dtype=bool)
    values = np.array([1.25, 2.5], dtype=np.float32)
    definition = {
        "geometry": {"maskT": manifest.encode("geometry/maskT", mask)},
        "forcing": {
            "constant": {},
            "series": {"uWind": manifest.encode("forcing/series/uWind", values)},
        },
    }
    path = tmp_path / "arrays.nc"
    with h5netcdf.File(path, "w") as file:
        manifest.write(file)
    with h5netcdf.File(path, "r") as file:
        restored = hydrate_definition(file, definition)
    np.testing.assert_array_equal(restored["geometry"]["maskT"], mask)
    assert restored["geometry"]["maskT"].dtype == mask.dtype
    np.testing.assert_array_equal(restored["forcing"]["series"]["uWind"], values)
    assert restored["forcing"]["series"]["uWind"].dtype == values.dtype


def test_manifest_rejects_external_links_and_unreferenced_arrays(
    tmp_path: Path,
) -> None:
    """Schema checks reject external data rather than following an HDF link."""
    manifest = ArrayManifest()
    reference = manifest.encode("forcing/series/uWind", np.arange(3.0))
    path = tmp_path / "arrays.nc"
    with h5netcdf.File(path, "w") as file:
        manifest.write(file)
    with (
        h5netcdf.File(path, "r") as file,
        pytest.raises(ValueError, match="unreferenced"),
    ):
        hydrate_definition(file, {})
    with h5py.File(path, "r+") as file:
        del file["definition/forcing/series/uWind"]
        file["definition/forcing/series/uWind"] = h5py.ExternalLink(
            "absent.nc", "/wind"
        )
    from veris.restart_arrays import validate_links

    with pytest.raises(ValueError, match="link"):
        validate_links(path)
    assert reference["array"].startswith("/definition/")


def test_manifest_preserves_non_native_definition_dtype_and_both_forcing_groups(
    tmp_path: Path,
) -> None:
    """Array descriptors preserve byte order and independent forcing sections."""
    manifest = ArrayManifest()
    constant = np.full((2, 2), 3, dtype=">f8")
    series = np.arange(3, dtype=np.float32)
    definition = {
        "geometry": None,
        "forcing": {
            "constant": {"uWind": manifest.encode("forcing/constant/uWind", constant)},
            "series": {"vWind": manifest.encode("forcing/series/vWind", series)},
        },
    }
    path = tmp_path / "both.nc"
    with h5netcdf.File(path, "w") as file:
        manifest.write(file)
    with h5netcdf.File(path, "r") as file:
        restored = hydrate_definition(file, definition)
    actual = restored["forcing"]["constant"]["uWind"]
    np.testing.assert_array_equal(actual, constant)
    assert actual.dtype == constant.dtype
    np.testing.assert_array_equal(restored["forcing"]["series"]["vWind"], series)
