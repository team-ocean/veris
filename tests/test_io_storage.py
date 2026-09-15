"""Real h5netcdf round trips and host-effect boundaries for Veris output."""

# ruff: noqa: DTZ001 -- model calendar dates are explicitly timezone-naive.

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray


def test_stream_records_have_staggered_dimensions_and_metadata(tmp_path: Path) -> None:
    from veris.io.storage import NetCDFWriter, read_record

    path = tmp_path / "fields.nc"
    fields = {"hIceMean": np.full((3, 4), 2.0), "uIce": np.full((3, 4), 6.0)}
    with NetCDFWriter(
        path, calendar="noleap", units="seconds since 2001-01-01 00:00:00"
    ) as writer:
        writer.append(
            "daily", fields, time=43200, bounds=(0, 86400), count=2, mean=True
        )
        writer.append(
            "daily",
            {k: v * 2 for k, v in fields.items()},
            time=129600,
            bounds=(86400, 172800),
            count=2,
            mean=True,
        )
    with h5netcdf.File(path) as file:
        group = file.groups["daily"]
        assert group.dimensions["time"].isunlimited()
        assert group.variables["uIce"].dimensions == ("time", "x_face", "y_center")
        assert group.variables["hIceMean"].attrs["units"] == "m"
        assert group.variables["uIce"].attrs["cell_methods"] == "time: mean"
        assert group.variables["time"].attrs["calendar"] == "noleap"
        np.testing.assert_array_equal(group.variables["sample_count"][:], [2, 2])
    record = read_record(path, stream="daily", index=1, variables=("uIce",))
    assert tuple(record.fields) == ("uIce",)
    np.testing.assert_array_equal(record.fields["uIce"], np.full((3, 4), 12.0))
    assert record.bounds == (86400, 172800)
    assert record.mean and record.count == 2
    with (
        pytest.raises(FileExistsError),
        NetCDFWriter(
            path, calendar="noleap", units="seconds since 2001-01-01"
        ) as writer,
    ):
        writer.append("instant", fields, time=0, bounds=(0, 0))


@pytest.mark.parametrize(
    "fields",
    [
        {"unknown": np.zeros((3, 4))},
        {"Area": np.zeros(4)},
        {"Area": np.zeros((3, 4)), "hIceMean": np.zeros((4, 3))},
        {"Area": np.array([["text"]])},
        {},
    ],
)
def test_bad_fields_fail_before_file_creation(
    tmp_path: Path, fields: dict[str, NDArray[Any]]
) -> None:
    from veris.io.storage import NetCDFWriter

    path = tmp_path / "bad.nc"
    with (
        NetCDFWriter(
            path, calendar="noleap", units="seconds since 2001-01-01"
        ) as writer,
        pytest.raises(ValueError),
    ):
        writer.append("instant", fields, time=0, bounds=(0, 0))
    assert not path.exists()


def test_full_snapshot_roundtrip_and_selected_physical_snapshot(tmp_path: Path) -> None:
    from veris.initialization import initialize
    from veris.io.storage import read_record, update_state, write_snapshot
    from veris.variables import VARIABLES

    state, conf, phys = initialize(
        settings_overrides={"nx": 3, "ny": 4, "use_sharding": False}
    )
    state = replace(state, hIceMean=jnp.arange(56.0).reshape(7, 8))
    path = tmp_path / "restart.nc"
    write_snapshot(
        path,
        state,
        elapsed=timedelta(days=2),
        start=datetime(2000, 1, 1),
        conf=conf,
        phys=phys,
    )
    record = read_record(path)
    assert set(record.fields) == set(VARIABLES)
    assert not record.mean
    assert record.time == 172800
    assert record.configuration["nx"] == 3
    assert record.physical_constants["rhoIce"] == phys.rhoIce
    target = replace(state, hIceMean=jnp.full((7, 8), -99.0))
    restored = update_state(target, record)
    expected = np.full((7, 8), -99.0)
    expected[2:-2, 2:-2] = np.asarray(state.hIceMean)[2:-2, 2:-2]
    np.testing.assert_array_equal(restored.hIceMean, expected)
    for name in VARIABLES:
        np.testing.assert_array_equal(
            getattr(restored, name)[2:-2, 2:-2], getattr(state, name)[2:-2, 2:-2]
        )
    physical = tmp_path / "physical.nc"
    write_snapshot(physical, state, variables=("hIceMean",))
    np.testing.assert_array_equal(
        read_record(physical).fields["hIceMean"],
        np.arange(56.0).reshape(7, 8)[2:-2, 2:-2],
    )
    restored = update_state(state, read_record(physical))
    np.testing.assert_array_equal(restored.hIceMean, state.hIceMean)


def test_reader_rejects_unknown_selection_and_mean_as_restart(tmp_path: Path) -> None:
    from veris.initialization import initialize
    from veris.io.storage import NetCDFWriter, read_record, update_state

    path = tmp_path / "mean.nc"
    with NetCDFWriter(
        path, calendar="noleap", units="seconds since 2001-01-01"
    ) as writer:
        writer.append(
            "daily",
            {"Area": np.ones((7, 8))},
            time=5,
            bounds=(0, 10),
            mean=True,
        )
    with pytest.raises(ValueError, match="variable"):
        read_record(path, stream="daily", variables=("theta",))
    state, _, _ = initialize(
        settings_overrides={"nx": 3, "ny": 4, "use_sharding": False}
    )
    with pytest.raises(ValueError, match="mean"):
        update_state(state, read_record(path, stream="daily"))


@pytest.mark.parametrize("transform", ["grad", "jvp", "jit"])
@pytest.mark.parametrize("closed_constant", [False, True])
def test_snapshot_refuses_transformed_calls_before_file_effects(
    tmp_path: Path, transform: str, closed_constant: bool
) -> None:
    from veris.io.storage import write_snapshot

    path = tmp_path / "forbidden.nc"
    constant = jnp.ones((6, 6))

    def objective(x: float | jax.Array) -> float | jax.Array:
        write_snapshot(path, {"Area": constant if closed_constant else constant * x})
        return x * x

    with pytest.raises(RuntimeError, match="transform"):
        if transform == "grad":
            jax.grad(objective)(2.0)
        elif transform == "jvp":
            jax.jvp(objective, (2.0,), (1.0,))
        else:
            jax.jit(objective)(2.0)
    assert not path.exists()


def test_final_snapshot_after_value_and_grad(tmp_path: Path) -> None:
    from veris.io.storage import read_record, write_snapshot

    def objective(x: float | jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        final = {"Area": jnp.ones((6, 6)) * x}
        return jnp.sum(final["Area"]), final

    (value, final), gradient = jax.value_and_grad(objective, has_aux=True)(2.0)
    assert value == 72 and gradient == 36
    write_snapshot(tmp_path / "final.nc", final)
    np.testing.assert_array_equal(
        read_record(tmp_path / "final.nc").fields["Area"], np.full((2, 2), 2.0)
    )


def test_writer_rejects_changed_shape_or_unsafe_dtype_before_extending(
    tmp_path: Path,
) -> None:
    from veris.io.storage import NetCDFWriter

    path = tmp_path / "types.nc"
    with NetCDFWriter(
        path, calendar="noleap", units="seconds since 2001-01-01"
    ) as writer:
        writer.append(
            "instant", {"Area": np.ones((3, 4), dtype="int32")}, time=0, bounds=(0, 0)
        )
        for value in [np.ones((4, 3), dtype="int32"), np.full((3, 4), 1.5)]:
            with pytest.raises(ValueError):
                writer.append("instant", {"Area": value}, time=1, bounds=(1, 1))
    with h5netcdf.File(path) as file:
        assert len(file.groups["instant"].dimensions["time"]) == 1
        np.testing.assert_array_equal(
            file.groups["instant"].variables["Area"][0], np.ones((3, 4))
        )


def test_reader_rejects_incompatible_variable_dimensions(tmp_path: Path) -> None:
    from veris.io import read_record

    path = tmp_path / "external.nc"
    with h5netcdf.File(path, "w") as file:
        group = file.create_group("snapshot")
        group.dimensions = {"time": 1, "wrong_x": 3, "y_center": 4}
        group.create_variable("Area", ("time", "wrong_x", "y_center"), dtype="f8")
    with pytest.raises(ValueError, match="dimensions"):
        read_record(path)


def test_snapshot_update_rejects_incompatible_initialized_shape(tmp_path: Path) -> None:
    from veris.initialization import initialize
    from veris.io import read_record, update_state, write_snapshot

    source, _, _ = initialize(
        settings_overrides={"nx": 3, "ny": 4, "use_sharding": False}
    )
    target, _, _ = initialize(
        settings_overrides={"nx": 4, "ny": 4, "use_sharding": False}
    )
    write_snapshot(tmp_path / "shape.nc", source, variables=("Area",))
    with pytest.raises(ValueError, match="shape"):
        update_state(target, read_record(tmp_path / "shape.nc"))


def test_snapshot_discards_nonuniform_storage_halos(tmp_path: Path) -> None:
    from veris.io import read_record, write_snapshot

    values = np.arange(72.0).reshape(8, 9)
    write_snapshot(tmp_path / "trimmed.nc", {"Area": values})
    record = read_record(tmp_path / "trimmed.nc")
    np.testing.assert_array_equal(record.fields["Area"], values[2:-2, 2:-2])


def test_writer_rejects_partial_mean_before_creating_file(tmp_path: Path) -> None:
    from veris.io.storage import NetCDFWriter

    path = tmp_path / "partial.nc"
    with (
        NetCDFWriter(
            path, calendar="noleap", units="seconds since 2001-01-01"
        ) as writer,
        pytest.raises(ValueError, match="partial"),
    ):
        writer.append(
            "mean",
            {"Area": np.ones((3, 4))},
            time=1,
            bounds=(0, 2),
            mean=True,
            partial=True,
        )
    assert not path.exists()


def test_float32_snapshot_preserves_settings_and_constants_metadata(
    tmp_path: Path,
) -> None:
    """NumPy float32 static parameters remain numeric JSON values in a snapshot."""
    from veris.initialization import initialize
    from veris.io import read_record, write_snapshot

    state, conf, phys = initialize(
        nx=3,
        ny=4,
        dtype="float32",
        settings_overrides={"use_sharding": False, "deltatTherm": 123.5},
        physical_overrides={"rhoAir": 1.25},
    )
    path = tmp_path / "float32.nc"
    write_snapshot(path, state, conf=conf, phys=phys)
    record = read_record(path)
    assert record.configuration["dtype"] == "float32"
    assert record.configuration["deltatTherm"] == 123.5
    assert record.physical_constants["rhoAir"] == 1.25
    assert record.physical_constants["longwaveCloudCoefficients"] == list(
        phys.longwaveCloudCoefficients
    )
    assert record.fields["hIceMean"].dtype == np.float32
