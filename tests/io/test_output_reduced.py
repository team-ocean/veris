"""Reduced output writes real records while keeping sampling modes exclusive."""

from datetime import timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import numpy as np
import pytest

from veris.io.configuration import OutputSettings, Stream
from veris.io.output import OutputManager
from veris.io.storage import ArrayFields, read_record


def _settings(*, enabled: bool = True) -> OutputSettings:
    return OutputSettings(
        enabled=enabled,
        streams=(
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
            Stream("instant", ("Area",), timedelta(seconds=1)),
        ),
    )


def _write(manager: OutputManager, value: Any, *, time: float = 2.0) -> None:
    manager.write_reduced(
        "mean",
        {"Area": value},
        time=time,
        bounds=(time - 2, time + 2),
        count=4,
        mean=True,
    )


def test_reduced_records_preserve_means_counts_bounds_and_instant_dtype(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reduced.nc"
    values = np.arange(72.0).reshape(8, 9)
    with OutputManager(path, _settings()) as manager:
        manager.begin_reduced()
        assert not path.exists()
        _write(manager, values)
        manager.write_reduced(
            "instant",
            {"Area": values.astype("float32")},
            time=4,
            bounds=(4, 4),
            count=1,
            mean=False,
        )
        assert manager.buffer_nbytes == 0
        manager.close(timedelta(seconds=100))
        manager.close()
    mean = read_record(path, stream="mean")
    instant = read_record(path, stream="instant")
    np.testing.assert_array_equal(mean.fields["Area"], values[2:-2, 2:-2])
    np.testing.assert_array_equal(instant.fields["Area"], values[2:-2, 2:-2])
    assert mean.count == 4 and mean.bounds == (0, 4) and mean.time == 2
    assert mean.mean and not mean.partial
    assert instant.count == 1 and instant.bounds == (4, 4) and not instant.mean
    assert instant.fields["Area"].dtype == np.float32
    with h5netcdf.File(path) as file:
        assert len(file.groups["mean"].dimensions["time"]) == 1


@pytest.mark.parametrize("writing_rank", [False, True])
def test_reduced_collector_receives_halos_once_and_nonwriter_creates_no_file(
    tmp_path: Path,
    writing_rank: bool,
) -> None:
    values = np.arange(72.0).reshape(8, 9)
    calls = []

    def collect(fields: Any, names: tuple[str, ...]) -> ArrayFields | None:
        calls.append(names)
        assert fields["Area"] is values
        return {"Area": values[2:-2, 2:-2]} if writing_rank else None

    path = tmp_path / "collected.nc"
    with OutputManager(path, _settings(), collector=collect) as manager:
        manager.begin_reduced()
        _write(manager, values)
    assert calls == [("Area",)]
    assert path.exists() == writing_rank
    if writing_rank:
        np.testing.assert_array_equal(
            read_record(path, stream="mean").fields["Area"],
            values[2:-2, 2:-2],
        )


def test_disabled_reduced_calls_have_no_collection_or_array_effects(
    tmp_path: Path,
) -> None:
    def collect(fields: Any, names: tuple[str, ...]) -> ArrayFields | None:
        raise AssertionError("disabled output collected arrays")

    path = tmp_path / "disabled.nc"
    with OutputManager(path, _settings(enabled=False), collector=collect) as manager:
        manager.begin_reduced()
        _write(manager, object())
        manager.sample({}, timedelta(seconds=-1))
        manager.close(timedelta(seconds=-1))
    assert not path.exists()


def test_reduced_requires_begin_and_rejects_direct_sampling(tmp_path: Path) -> None:
    with OutputManager(tmp_path / "modes.nc", _settings()) as manager:
        with pytest.raises(RuntimeError, match="reduced"):
            _write(manager, np.ones((6, 6)))
        manager.begin_reduced()
        with pytest.raises(RuntimeError, match="reduced"):
            manager.sample({"Area": np.ones((6, 6))}, timedelta(0))
        with pytest.raises(RuntimeError, match="reduced"):
            manager.begin_reduced()


def test_successful_direct_sample_prevents_reduced_mode(tmp_path: Path) -> None:
    with OutputManager(tmp_path / "direct.nc", _settings()) as manager:
        manager.sample({"Area": np.ones((6, 6))}, timedelta(0))
        with pytest.raises(RuntimeError, match="sampl"):
            manager.begin_reduced()


def test_failed_direct_preflight_does_not_claim_manager(tmp_path: Path) -> None:
    with OutputManager(tmp_path / "retry.nc", _settings()) as manager:
        with pytest.raises(ValueError, match="missing"):
            manager.sample({}, timedelta(0))
        manager.begin_reduced()
        _write(manager, np.ones((6, 6)))


@pytest.mark.parametrize("reduced", [False, True])
def test_closed_manager_rejects_reduced_operations(
    tmp_path: Path, reduced: bool
) -> None:
    manager = OutputManager(tmp_path / "closed.nc", _settings())
    if reduced:
        manager.begin_reduced()
    manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        manager.begin_reduced()
    with pytest.raises(RuntimeError, match="closed"):
        _write(manager, np.ones((6, 6)))


def test_reduced_schema_failure_allows_retry_without_extra_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "retry-schema.nc"
    with OutputManager(path, _settings()) as manager:
        manager.begin_reduced()
        _write(manager, np.ones((6, 7)))
        with pytest.raises(ValueError, match="shape"):
            _write(manager, np.ones((7, 7)), time=6)
        _write(manager, np.full((6, 7), 3.0), time=6)
    with h5netcdf.File(path) as file:
        assert len(file.groups["mean"].dimensions["time"]) == 2
    assert read_record(path, stream="mean").time == 6


@pytest.mark.parametrize("name, mean", [("unknown", True), ("instant", True)])
def test_reduced_rejects_unconfigured_stream_or_record_kind(
    tmp_path: Path, name: str, mean: bool
) -> None:
    path = tmp_path / "invalid-stream.nc"
    with OutputManager(path, _settings()) as manager:
        manager.begin_reduced()
        with pytest.raises(ValueError, match="stream|kind"):
            manager.write_reduced(
                name,
                {"Area": np.ones((6, 7))},
                time=2,
                bounds=(0, 4),
                count=4,
                mean=mean,
            )
    assert not path.exists()


def test_reduced_exception_closes_file_and_preserves_written_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exception.nc"
    manager = OutputManager(path, _settings())
    with pytest.raises(ValueError, match="integration failed"), manager:
        manager.begin_reduced()
        _write(manager, np.ones((6, 7)))
        raise ValueError("integration failed")
    assert read_record(path, stream="mean").count == 4
    with pytest.raises(RuntimeError, match="closed"):
        _write(manager, np.ones((6, 7)), time=6)
