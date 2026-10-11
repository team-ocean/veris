"""Lifecycle output preserves absolute sampling and unfinished device means.

Expected wind fields come from independent pre/post-step index arithmetic. No production
output planner, reduction helper or physics implementation builds the oracle.
"""

# ruff: noqa: DTZ001 -- model calendar dates are explicitly timezone-naive.

import json
from dataclasses import fields, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import numpy as np
import pytest

import veris
from veris.io import FixedDate, OutputSettings, Stream, read_record


def _definition(
    *,
    dtype: str = "float32",
    step_seconds: float = 1,
    output: OutputSettings | None = None,
    steps: int = 8,
) -> Any:
    """Small real growth preset with easily checked prescribed wind values."""
    return veris.Experiment.builtin(
        "growth",
        grid=veris.Grid(2, 2),
        settings={"dtype": dtype, "deltatTherm": step_seconds},
        forcing=veris.Forcing(series={"uWind": np.arange(3.0, 3.0 + steps)}),
        steps=steps,
        output=output,
    )


def _output(*, initial: bool = True) -> OutputSettings:
    """Independent instantaneous and four-second mean streams."""
    return OutputSettings(
        streams=(
            Stream("instant", ("uWind",), timedelta(seconds=2)),
            Stream("mean", ("uWind",), timedelta(seconds=1), timedelta(seconds=4)),
        ),
        sample_initial=initial,
    )


def _assert_same_state(actual: Any, expected: Any) -> None:
    """Report concise all-field differences rather than dumping large arrays."""
    for item in fields(expected):
        left, right = (
            np.asarray(getattr(actual, item.name)),
            np.asarray(getattr(expected, item.name)),
        )
        assert left.dtype == right.dtype, f"ERROR {item.name} dtype changed"
        assert np.array_equal(left, right), (
            f"ERROR {item.name} differs, max_abs={np.max(np.abs(left - right)):.6g}"
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("initial", [False, True])
def test_split_runs_preserve_absolute_samples_means_and_every_state_field(
    tmp_path: Path, dtype: str, initial: bool
) -> None:
    """A split inside each mean window must retain sums and avoid duplicates."""
    definition = _definition(dtype=dtype, output=_output(initial=initial))
    whole_path, split_path = tmp_path / "whole.nc", tmp_path / "split.nc"
    with veris.Simulation.initialize(definition, output_path=whole_path) as whole:
        uninterrupted = whole.run()
        assert uninterrupted.observations is None
    with veris.Simulation.initialize(definition, output_path=split_path) as split:
        split.run(0)
        split.run(3)
        split.run(0)
        split.run(2)
        final = split.run()
        split.run(0)
        assert final.elapsed == timedelta(seconds=8)
        assert final.observations is None
    _assert_same_state(final.state, uninterrupted.state)
    with (
        h5netcdf.File(whole_path, "r") as whole_file,
        h5netcdf.File(split_path, "r") as split_file,
    ):
        assert set(whole_file.groups) == set(split_file.groups) == {"instant", "mean"}
        for group_name in whole_file.groups:
            whole_group, split_group = (
                whole_file.groups[group_name],
                split_file.groups[group_name],
            )
            assert set(whole_group.variables) == set(split_group.variables)
            for name in whole_group.variables:
                np.testing.assert_array_equal(
                    split_group.variables[name][:], whole_group.variables[name][:]
                )
        instant, mean = split_file.groups["instant"], split_file.groups["mean"]
        np.testing.assert_array_equal(
            instant.variables["time"][:], [0, 2, 4, 6, 8] if initial else [2, 4, 6, 8]
        )
        np.testing.assert_array_equal(
            instant.variables["uWind"][:, 0, 0],
            [3, 4, 6, 8, 10] if initial else [4, 6, 8, 10],
        )
        np.testing.assert_array_equal(
            mean.variables["time_bounds"][:], [[0, 4], [4, 8]]
        )
        np.testing.assert_array_equal(
            mean.variables["sample_count"][:], [4 if initial else 3, 4]
        )
        np.testing.assert_array_equal(
            mean.variables["uWind"][:, 0, 0], [3.75 if initial else 4, 7.5]
        )
        assert instant.variables["uWind"].dtype == np.dtype(dtype)
        assert mean.variables["uWind"].dtype == np.dtype("float64")
        assert mean.variables["time"].dtype == np.dtype("float64")


def test_pending_mean_completes_without_another_due_sample(tmp_path: Path) -> None:
    """The resumed planner must retain an initial-only window until its boundary."""
    output = OutputSettings(
        streams=(
            Stream("mean", ("uWind",), timedelta(seconds=10), timedelta(seconds=5)),
        )
    )
    path = tmp_path / "sparse.nc"
    with veris.Simulation.initialize(
        _definition(output=output, steps=6), output_path=path
    ) as simulation:
        simulation.run(2)
        assert not path.exists()
        simulation.run(4)
    record = read_record(path, stream="mean")
    assert record.bounds == (0, 5) and record.count == 1
    np.testing.assert_array_equal(record.fields["uWind"], 3)


def test_split_non_aligned_windows_keep_exact_bounds_and_pretransition_samples(
    tmp_path: Path,
) -> None:
    """A five-second window completes at t=6 for two-second physics steps."""
    output = OutputSettings(
        streams=(
            Stream("mean", ("uWind",), timedelta(seconds=2), timedelta(seconds=5)),
        )
    )
    path = tmp_path / "nonaligned.nc"
    with veris.Simulation.initialize(
        _definition(step_seconds=2, output=output, steps=5), output_path=path
    ) as simulation:
        simulation.run(2)
        simulation.run(1)
        simulation.run(2)
    with h5netcdf.File(path, "r") as file:
        group = file.groups["mean"]
        np.testing.assert_array_equal(
            group.variables["time_bounds"][:], [[0, 5], [5, 10]]
        )
        np.testing.assert_array_equal(group.variables["sample_count"][:], [3, 2])
        np.testing.assert_array_equal(group.variables["time"][:], [2.5, 7.5])
        # XLA's existing reciprocal division can differ from host division by
        # one float64 ULP. Sampling, counts and bounds remain exact above.
        np.testing.assert_array_max_ulp(
            group.variables["uWind"][:, 0, 0], np.array([10 / 3, 5.5]), maxulp=1
        )


@pytest.mark.parametrize("fixed", [False, True])
def test_calendar_origin_survives_run_splits_and_history_saves_resolved_metadata(
    tmp_path: Path, fixed: bool
) -> None:
    """Starting at noon discards only the original incomplete calendar window."""
    origin = FixedDate(2000, 2, 30, hour=12) if fixed else datetime(2000, 2, 29, 12)
    output = OutputSettings(
        start=origin,
        calendar="360_day" if fixed else "gregorian",
        streams=(Stream("daily", ("uWind",), timedelta(hours=6), "daily"),),
    )
    definition = _definition(step_seconds=21600, output=output, steps=8)
    path = tmp_path / "calendar.nc"
    with veris.Simulation.initialize(definition, output_path=path) as simulation:
        simulation.run(3)
        simulation.run(2)
        simulation.run(3)
    record = read_record(path, stream="daily")
    assert record.bounds == (43200, 129600) and record.count == 4
    np.testing.assert_array_equal(record.fields["uWind"], 5.5)
    expected_day = "2000-02-30" if fixed else "2000-02-29"
    assert record.units == f"seconds since {expected_day} 12:00:00"
    assert record.calendar == ("360_day" if fixed else "proleptic_gregorian")
    assert record.configuration["deltatTherm"] == 21600
    assert record.physical_constants["rhoSnow"] == definition.physical_constants.rhoSnow
    with h5netcdf.File(path, "r") as file:
        assert json.loads(file.attrs["experiment"]) == definition.to_dict()


def test_snapshot_uses_established_group_precision_and_absolute_model_time(
    tmp_path: Path,
) -> None:
    """Visualization snapshots stay selected inputs rather than complete restarts."""
    definition = _definition(steps=3)
    path = tmp_path / "snapshot.nc"
    with veris.Simulation.initialize(definition) as simulation:
        final = simulation.run(3)
        simulation.snapshot(path, variables=("hIceMean", "Area"))
    record = read_record(path)
    assert set(record.fields) == {"hIceMean", "Area"}
    assert record.time == 3 and record.bounds == (3, 3) and not record.mean
    assert record.configuration["dtype"] == "float32"
    for name, array in record.fields.items():
        assert array.shape == (2, 2) and array.dtype == np.dtype("float32")
        np.testing.assert_array_equal(
            array, np.asarray(getattr(final.state, name))[2:-2, 2:-2]
        )
    with h5netcdf.File(path, "r") as file:
        assert json.loads(file.attrs["experiment"]) == definition.to_dict()


def test_selected_input_refreshes_ice_derivatives_and_halos_without_resetting_time(
    tmp_path: Path,
) -> None:
    """Selected input replaces only requested physical fields and derived ice data."""
    donor = replace(_definition(dtype="float64"), initial_conditions={"hIceMean": 2})
    path = tmp_path / "ice.nc"
    with veris.Simulation.initialize(donor) as simulation:
        simulation.snapshot(path, variables=("hIceMean",))
    definition = _definition(dtype="float64")
    with veris.Simulation.initialize(definition) as simulation:
        before = simulation.run(2)
        simulation.load_fields(path, ("hIceMean",))
        after = simulation.run(0)
        assert before.elapsed == after.elapsed == timedelta(seconds=2)
        np.testing.assert_array_equal(after.state.hIceMean, 2)
        np.testing.assert_array_equal(after.state.uWind, before.state.uWind)
        phys = definition.physical_constants
        np.testing.assert_array_equal(
            after.state.SeaIceLoad,
            2 * phys.rhoIce + phys.rhoSnow * after.state.hSnowMean,
        )
        np.testing.assert_allclose(
            after.state.recip_hIceMean, 1 / np.sqrt(4 + phys.hIce_reg), rtol=1e-14
        )
        final = simulation.run(1)
        assert final.elapsed == timedelta(seconds=3)
        np.testing.assert_array_equal(final.state.uWind, 5)


def test_enabled_output_requires_a_path_and_rejects_existing_files(
    tmp_path: Path,
) -> None:
    """Host controls must fail before overwriting an existing experiment artifact."""
    definition = _definition(output=_output())
    with pytest.raises(ValueError, match="output.*path|path.*output"):
        veris.Simulation.initialize(definition)
    path = tmp_path / "existing.nc"
    path.write_bytes(b"existing experiment artifact")
    with pytest.raises(FileExistsError):
        veris.Simulation.initialize(definition, output_path=path)
    assert path.read_bytes() == b"existing experiment artifact"


def test_unreachable_sampling_fails_without_creating_a_file(tmp_path: Path) -> None:
    output = OutputSettings(
        streams=(Stream("ice", ("hIceMean",), timedelta(seconds=3)),)
    )
    path = tmp_path / "unreachable.nc"
    with (
        pytest.raises(ValueError, match="sampling|multiple|timestep"),
        veris.Simulation.initialize(
            _definition(step_seconds=2, output=output), output_path=path
        ) as simulation,
    ):
        simulation.run(2)
    assert not path.exists()


def test_disabled_output_needs_no_path_and_no_observation_history(
    tmp_path: Path,
) -> None:
    definition = _definition(output=replace(_output(), enabled=False))
    path = tmp_path / "disabled.nc"
    with veris.Simulation.initialize(definition, output_path=path) as simulation:
        result = simulation.run(2)
        assert result.observations is None
    assert not path.exists()


def test_context_failure_closes_writer_and_final_partial_mean_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed host application must release its already-created netCDF writer."""
    from veris.io import storage

    closed = []
    original_close = storage.NetCDFWriter.close

    def close(writer: Any) -> None:
        original_close(writer)
        closed.append(writer.path)

    monkeypatch.setattr(storage.NetCDFWriter, "close", close)
    path = tmp_path / "failure.nc"
    with (
        pytest.raises(RuntimeError, match="application failed"),
        veris.Simulation.initialize(
            _definition(output=_output()), output_path=path
        ) as simulation,
    ):
        simulation.run(3)
        raise RuntimeError("application failed")
    assert path in closed
    with h5netcdf.File(path, "r") as file:
        assert "mean" not in file.groups
        np.testing.assert_array_equal(
            file.groups["instant"].variables["time"][:], [0, 2]
        )
    simulation.close()
    with pytest.raises(RuntimeError, match="closed"):
        simulation.run(1)
