"""Complete public restarts preserve evolved fields, time and forcing.

Independent real trajectories compare every numerical field and halo on one
backend, covering version 1 schema, clocks, corruption and output continuation.
"""

# ruff: noqa: DTZ001 -- model calendar dates intentionally have no timezone.

import json
import os
import subprocess
import sys
from dataclasses import fields, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import h5py
import jax
import numpy as np
import pytest

import veris
from veris.io import (
    FixedDate,
    OutputManager,
    OutputSettings,
    Stream,
    read_record,
    write_snapshot,
)


def _assert_same_state(actual: Any, expected: Any) -> None:
    """Compare all array leaves including physical wall faces and outer halos."""
    assert tuple(item.name for item in fields(actual)) == tuple(
        item.name for item in fields(expected)
    )
    for item in fields(expected):
        left, right = (
            np.asarray(getattr(actual, item.name)),
            np.asarray(getattr(expected, item.name)),
        )
        assert left.shape == right.shape, f"ERROR restart shape: {item.name}"
        assert left.dtype == right.dtype, f"ERROR restart dtype: {item.name}"
        assert np.isfinite(left).all() and np.isfinite(right).all(), (
            f"ERROR restart nonfinite: {item.name}"
        )
        assert np.array_equal(left, right), (
            f"ERROR restart mismatch: {item.name}; "
            f"max_abs={np.max(np.abs(left - right)):.6g}"
        )


def _experiment(
    preset: str = "island",
    *,
    dtype: str = "float64",
    cyclic: bool = True,
    steps: int = 6,
    seconds: float = 600,
    output: OutputSettings | None = None,
) -> Any:
    """Small rectangular real physics with distinct forcing at every step."""
    nx, ny = (2, 2) if preset == "growth" else (4, 6)
    x, y = np.arange(nx)[:, None], np.arange(ny)[None, :]
    pattern = 0.08 * x + 0.04 * y
    indices = np.arange(steps, dtype=np.float64)[:, None, None]
    series = {
        "uWind": 2 + pattern[None, :, :] + 0.4 * indices,
        "vWind": -0.3 + 0.5 * pattern[None, :, :] - 0.1 * indices,
        "theta": 270 + 0.2 * np.arange(steps, dtype=np.float64),
    }
    if preset == "island":
        series["Qnet"] = 65 + 13 * indices + pattern[None, :, :]
    return veris.Experiment.builtin(
        preset,
        grid=veris.Grid(nx, ny),
        settings={
            "dtype": dtype,
            "enable_cyclic_y": cyclic,
            "noSlip": True,
            "nEVPsteps": 3,
            "deltatDyn": seconds,
            "deltatTherm": seconds,
        },
        constants={"rhoAir": 1.25, "rhoSnow": 335},
        initial_conditions={
            "hIceMean": 0.7 + pattern,
            "hSnowMean": 0.05,
            "Area": 0.85,
            "uIce": 0.008 + 0.002 * pattern,
            "vIce": -0.003 + 0.001 * pattern,
        },
        forcing=veris.Forcing(constant={"ocSalt": 34}, series=series),
        execution=veris.Execution(backend="cpu"),
        steps=steps,
        output=output,
    )


@pytest.mark.parametrize(
    "preset,dtype,cyclic",
    [
        ("island", "float64", True),
        ("island", "float64", False),
        ("dynamics", "float64", True),
        ("dynamics", "float64", False),
        ("growth", "float64", True),
        ("growth", "float64", False),
        ("island", "float32", False),
        ("growth", "float32", True),
    ],
)
@pytest.mark.parametrize("checkpoint_step", [1, 3])
def test_complete_restart_matches_independent_uninterrupted_trajectory(
    tmp_path: Path, preset: str, dtype: str, cyclic: bool, checkpoint_step: int
) -> None:
    """Forcing must continue at the absolute cursor; all evolved fields survive."""
    definition = _experiment(preset, dtype=dtype, cyclic=cyclic)
    path = tmp_path / "restart.nc"
    with veris.Simulation.initialize(definition) as reference:
        initial = reference.state
        reference.run()
        expected = reference.state
        assert not np.array_equal(initial.hIceMean, expected.hIceMean) or not (
            np.array_equal(initial.uIce, expected.uIce)
        ), "ERROR restart reference must exercise real evolution"
    with veris.Simulation.initialize(definition) as first:
        first.run(checkpoint_step)
        checkpoint = first.state
        first.save_restart(path)
        assert first.completed_steps == checkpoint_step
        assert first.elapsed == checkpoint_step * definition.step_duration
    with veris.Simulation.from_restart(path) as resumed:
        _assert_same_state(resumed.state, checkpoint)
        assert resumed.configuration == definition.configuration
        assert resumed.physical_constants == definition.physical_constants
        assert resumed.experiment.to_dict() == definition.to_dict()
        assert resumed.completed_steps == checkpoint_step
        assert resumed.elapsed == checkpoint_step * definition.step_duration
        resumed.run(1)
        resumed.run()
        assert resumed.completed_steps == definition.steps
        assert resumed.elapsed == definition.steps * definition.step_duration
        _assert_same_state(resumed.state, expected)


def test_no_slip_north_wall_traction_survives_before_halo_refresh_and_next_step(
    tmp_path: Path,
) -> None:
    """The first north sigma12 halo stores a physical stress, not redundant data."""
    # The dynamics preset's last y row is land and cannot carry wall traction.
    # Coupled island geometry has wet north faces for the dynamics stress stage.
    definition = _experiment("island", cyclic=False, steps=3)
    path = tmp_path / "wall.nc"
    with veris.Simulation.initialize(definition) as original:
        original.step()
        checkpoint = original.state
        traction = np.asarray(checkpoint.sigma12)[2:-2, -2].copy()
        assert np.any(traction != 0), "ERROR wall oracle needs nonzero north traction"
        original.save_restart(path)
        original.step()
        expected = original.state
    with veris.Simulation.from_restart(path) as resumed:
        np.testing.assert_array_equal(
            np.asarray(resumed.state.sigma12)[2:-2, -2], traction
        )
        _assert_same_state(resumed.state, checkpoint)
        assert np.all(np.asarray(resumed.state.sigma12)[:, -1] == 0)
        resumed.step()
        _assert_same_state(resumed.state, expected)


def test_float32_subsecond_restart_preserves_exact_clock_and_series_position(
    tmp_path: Path,
) -> None:
    """Normalized float32 settings must retain microsecond time on restoration."""
    definition = _experiment("growth", dtype="float32", seconds=0.100001, steps=5)
    with veris.Simulation.initialize(definition) as whole:
        expected = whole.run()
    checkpoint = tmp_path / "float32-clock.nc"
    with veris.Simulation.initialize(definition) as split:
        assert split.run(2).elapsed == timedelta(microseconds=200002)
        split.save_restart(checkpoint)
    with veris.Simulation.from_restart(checkpoint) as resumed:
        assert resumed.completed_steps == 2
        assert resumed.elapsed == timedelta(microseconds=200002)
        actual = resumed.run()
        assert actual.steps == 3 and actual.completed_steps == 5
        assert actual.elapsed == expected.elapsed == timedelta(microseconds=500005)
        _assert_same_state(actual.state, expected.state)


def test_restart_save_preserves_live_state_cursor_and_file_ownership(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.nc"
    definition = _experiment("growth", seconds=1.000001, steps=4)
    with veris.Simulation.initialize(definition) as simulation:
        simulation.run(2)
        state, elapsed = simulation.state, simulation.elapsed
        simulation.save_restart(path)
        _assert_same_state(simulation.state, state)
        assert simulation.completed_steps == 2
        assert elapsed == simulation.elapsed == timedelta(microseconds=2_000_002)
        saved_bytes = path.read_bytes()
        with pytest.raises(FileExistsError):
            simulation.save_restart(path)
        assert path.read_bytes() == saved_bytes
        simulation.step()
        expected = simulation.state
    with veris.Simulation.from_restart(path) as resumed:
        assert resumed.elapsed == timedelta(microseconds=2_000_002)
        assert resumed.completed_steps == 2
        resumed.step()
        _assert_same_state(resumed.state, expected)
    # Closed h5netcdf handles must permit reopening and removal on completion.
    with h5netcdf.File(path, "r") as file:
        assert file.attrs["veris_file_kind"] == "restart"
        assert file.attrs["veris_restart_version"] == 1
        clock = json.loads(file.attrs["clock"])
        assert clock == {
            "completed_steps": 2,
            "elapsed_microseconds": 2_000_002,
            "forcing_step": 2,
        }
        assert {item.name for item in fields(state)} <= file.groups[
            "state"
        ].variables.keys()
        assert file.groups["state"].variables["hIceMean"].shape == (1, 6, 6)
    path.unlink()


def test_checkpoint_at_exhausted_forcing_restores_without_selecting_next_sample(
    tmp_path: Path,
) -> None:
    path = tmp_path / "finished.nc"
    definition = _experiment("growth", steps=2)
    with veris.Simulation.initialize(definition) as original:
        original.run()
        original.save_restart(path)
        expected = original.state
    with veris.Simulation.from_restart(path) as resumed:
        resumed.run(0)
        _assert_same_state(resumed.state, expected)
        assert resumed.completed_steps == 2
        with pytest.raises(ValueError, match="forcing|exhaust"):
            resumed.step()
        _assert_same_state(resumed.state, expected)
        assert resumed.completed_steps == 2


def _geometry(nx: int, ny: int) -> Any:
    """Nonuniform global physical geometry, including boolean volume masks."""
    x, y = np.arange(nx)[:, None], np.arange(ny)[None, :]
    mask = np.ones((nx, ny, 2), dtype=bool)
    mask[1, 2, -1] = False
    area = 8000.0 * 9000.0 * (1 + 0.01 * x + 0.02 * y)
    return veris.OceanGeometry(
        maskT=mask,
        maskU=mask.copy(),
        maskV=mask.copy(),
        ht=-80.0 - x - 2 * y,
        coriolis_t=np.broadcast_to(1e-4 + 1e-6 * y, (nx, ny)),
        dxt=8000.0 + 10 * np.arange(nx),
        dxu=8050.0 + 11 * np.arange(nx),
        dyt=9000.0 + 12 * np.arange(ny),
        dyu=9060.0 + 13 * np.arange(ny),
        area_t=area,
        area_u=area * 1.01,
        area_v=area * 1.02,
    )


def test_custom_geometry_and_large_forcing_use_numeric_groups_and_owned_inputs(
    tmp_path: Path,
) -> None:
    """Resolved metadata must not embed large grid/time arrays into JSON attrs."""
    nx, ny, steps = 4, 5, 257
    x, y = np.arange(nx)[:, None], np.arange(ny)[None, :]
    wind = 2 + np.arange(steps)[:, None, None] * 0.01 + 0.1 * x + 0.02 * y
    definition = veris.Experiment(
        grid=veris.Grid(nx, ny),
        geometry=_geometry(nx, ny),
        physics="thermodynamics",
        settings={"dtype": "float32", "deltatTherm": 600},
        initial_conditions={"hIceMean": 0.6 + 0.03 * x + 0.01 * y, "Area": 0.8},
        forcing=veris.Forcing(
            constant={"theta": 270.0, "Qnet": 80.0, "ocSalt": 34.0},
            series={"uWind": wind, "vWind": -0.2 * wind},
        ),
        execution=veris.Execution(backend="cpu"),
        steps=steps,
    )
    path = tmp_path / "custom.nc"
    with veris.Simulation.initialize(definition) as simulation:
        simulation.run(2)
        checkpoint = simulation.state
        simulation.save_restart(path)
    with h5netcdf.File(path, "r") as file:
        encoded = file.attrs["experiment"]
        manifest = json.loads(encoded)
        assert len(encoded) < 25_000, "ERROR restart embeds numerical arrays in JSON"
        reference = manifest["forcing"]["series"]["uWind"]
        assert reference["array"] == "/definition/forcing/series/uWind"
        assert reference["shape"] == [steps, nx, ny]
        series = file.groups["definition"].groups["forcing"].groups["series"]
        np.testing.assert_array_equal(series.variables["uWind"][:], wind)
        assert manifest["geometry"]["maskT"]["dtype"] == "bool"
    with veris.Simulation.from_restart(path) as resumed:
        _assert_same_state(resumed.state, checkpoint)
        assert resumed.experiment.to_dict() == definition.to_dict()
        assert resumed.experiment.geometry is not None
        assert resumed.experiment.geometry.maskT.dtype == np.dtype(bool)
        assert resumed.experiment.forcing.series["uWind"].shape == (steps, nx, ny)
        np.testing.assert_array_equal(
            resumed.experiment.forcing.at(2)["uWind"], wind[2]
        )
        with pytest.raises((AttributeError, ValueError), match="immutable|read.only"):
            resumed.experiment.forcing.series["uWind"].resize((1, nx, ny))
        with pytest.raises(ValueError, match="read.only"):
            resumed.experiment.geometry.maskT[0, 0, 0] = False


@pytest.mark.parametrize("fixed", [False, True])
def test_resume_discards_crossing_mean_and_preserves_absolute_output_calendar(
    tmp_path: Path, fixed: bool
) -> None:
    """[0,4) is incomplete after restart at 3; [4,8) is a complete new mean."""
    origin = (
        FixedDate(2000, 2, 30, microsecond=123456)
        if fixed
        else datetime(2000, 2, 29, microsecond=123456)
    )
    output = OutputSettings(
        start=origin,
        calendar="360_day" if fixed else "gregorian",
        streams=(
            Stream("instant", ("hIceMean",), timedelta(seconds=2)),
            Stream("mean", ("hIceMean",), timedelta(seconds=1), timedelta(seconds=4)),
        ),
    )
    definition = _experiment("growth", steps=8, seconds=1, output=output)
    path = tmp_path / "checkpoint.nc"
    original_output, resumed_output = tmp_path / "original.nc", tmp_path / "resumed.nc"
    # Independent real states establish the arithmetic mean without I/O scheduling.
    total = np.zeros((2, 2), dtype=np.float64)
    with veris.Simulation.initialize(replace(definition, output=None)) as reference:
        for index in range(8):
            if 4 <= index < 8:
                total += np.asarray(reference.state.hIceMean)[2:-2, 2:-2]
            reference.step()
        expected = reference.state
    with veris.Simulation.initialize(definition, output_path=original_output) as first:
        first.run(3)
        first.save_restart(path)
    original_bytes = original_output.read_bytes()
    with veris.Simulation.from_restart(path, output_path=resumed_output) as resumed:
        assert resumed.experiment.output == output
        assert resumed.elapsed == timedelta(seconds=3)
        resumed.run(2)
        resumed.run()
        _assert_same_state(resumed.state, expected)
    assert original_output.read_bytes() == original_bytes
    with h5netcdf.File(resumed_output, "r") as file:
        instant, mean = file.groups["instant"], file.groups["mean"]
        np.testing.assert_array_equal(instant.variables["time"][:], [4, 6, 8])
        np.testing.assert_array_equal(mean.variables["time_bounds"][:], [[4, 8]])
        np.testing.assert_array_equal(mean.variables["sample_count"][:], [4])
        assert mean.variables["hIceMean"].dtype == np.dtype("float64")
        assert not np.any(mean.variables["partial"][:])
    record = read_record(resumed_output, stream="mean")
    np.testing.assert_array_equal(record.fields["hIceMean"], total / 4)
    assert record.time == 6 and record.bounds == (4, 8)
    assert record.calendar == ("360_day" if fixed else "proleptic_gregorian")
    assert record.units.endswith("00:00:00.123456")


@pytest.fixture
def output_restart(tmp_path: Path) -> tuple[Path, Path, OutputSettings]:
    """A checkpoint with enabled output anchored at a subsecond Gregorian origin."""
    output = OutputSettings(
        start=datetime(2000, 1, 1, microsecond=71),
        streams=(Stream("ice", ("hIceMean",), timedelta(seconds=1)),),
    )
    checkpoint, old = tmp_path / "restart.nc", tmp_path / "old-output.nc"
    with veris.Simulation.initialize(
        _experiment("growth", steps=4, seconds=1, output=output), output_path=old
    ) as simulation:
        simulation.run(1)
        simulation.save_restart(checkpoint)
    return checkpoint, old, output


def test_enabled_restart_requires_new_output_and_allows_stream_policy_override(
    tmp_path: Path, output_restart: tuple[Path, Path, OutputSettings]
) -> None:
    checkpoint, old, saved = output_restart
    old_bytes, checkpoint_bytes = old.read_bytes(), checkpoint.read_bytes()
    with pytest.raises(ValueError, match="output_path|output path"):
        veris.Simulation.from_restart(checkpoint)
    for destination in (old, checkpoint):
        with pytest.raises(FileExistsError):
            veris.Simulation.from_restart(checkpoint, output_path=destination)
    assert old.read_bytes() == old_bytes
    assert checkpoint.read_bytes() == checkpoint_bytes
    override = replace(
        saved,
        streams=(Stream("area", ("Area",), timedelta(seconds=1)),),
        sample_initial=False,
    )
    destination = tmp_path / "changed-streams.nc"
    with veris.Simulation.from_restart(
        checkpoint, output_path=destination, output_settings=override
    ) as resumed:
        assert resumed.experiment.output == override
        resumed.run()
    with h5netcdf.File(destination, "r") as file:
        assert set(file.groups) == {"area"}
        assert (
            file.groups["area"]
            .variables["time"]
            .attrs["units"]
            .endswith("00:00:00.000071")
        )


@pytest.mark.parametrize("change", ["origin", "calendar"])
def test_restart_output_override_cannot_change_time_semantics(
    tmp_path: Path, output_restart: tuple[Path, Path, OutputSettings], change: str
) -> None:
    checkpoint, _, saved = output_restart
    override = (
        replace(saved, start=datetime(2000, 1, 1, microsecond=72))
        if change == "origin"
        else replace(
            saved, start=FixedDate(2000, 1, 1, microsecond=71), calendar="noleap"
        )
    )
    destination = tmp_path / "invalid-time.nc"
    with pytest.raises(ValueError, match="calendar|origin|start"):
        veris.Simulation.from_restart(
            checkpoint, output_path=destination, output_settings=override
        )
    assert not destination.exists()


@pytest.fixture
def restart_path(tmp_path: Path) -> Path:
    """A complete CPU file for isolated corruption checks and cold read preflight."""
    path = tmp_path / "valid.nc"
    with veris.Simulation.initialize(_experiment("dynamics", steps=4)) as simulation:
        simulation.run(2)
        simulation.save_restart(path)
    return path


def _edit_json_attribute(path: Path, name: str, edit: Any) -> None:
    """Change one declared metadata value without damaging the surrounding file."""
    with h5netcdf.File(path, "r+") as file:
        value = json.loads(file.attrs[name])
        edit(value)
        file.attrs[name] = json.dumps(value, allow_nan=False)


def _replace_state_variable(path: Path, *, wrong_dtype: bool = False) -> None:
    """Make a real invalid variable while retaining valid netCDF dimension scales."""
    with h5netcdf.File(path, "r") as file:
        variable = file.groups["state"].variables["hIceMean"]
        values = np.asarray(variable[:])
        dimensions, attributes = variable.dimensions, dict(variable.attrs)
    with h5py.File(path, "r+") as file:
        # Remove references from dimension scales before deleting the dataset.
        # A dangling unlimited-time reference prevents creation of the intended
        # corrupt variable and never reaches the restart validation under test.
        for dimension in file["state/hIceMean"].dims:
            for scale in tuple(dimension.values()):
                dimension.detach_scale(scale)
        del file["state/hIceMean"]
    with h5netcdf.File(path, "r+") as file:
        group = file.groups["state"]
        if wrong_dtype:
            values = values.astype(np.float32)
        else:
            group.dimensions["corrupt_x"] = values.shape[1] + 1
            dimensions = (dimensions[0], "corrupt_x", dimensions[2])
            values = np.pad(values, ((0, 0), (0, 1), (0, 0)), mode="edge")
        variable = group.create_variable("hIceMean", dimensions, dtype=values.dtype)
        variable.attrs.update(attributes)
        variable[:] = values


def _corrupt(path: Path, problem: str) -> None:
    """Each alteration targets one validated restart invariant."""
    if problem in {"missing_field", "nonfinite", "unknown_field"}:
        with h5netcdf.File(path, "r+") as file:
            if problem == "nonfinite":
                file.groups["state"].variables["hIceMean"][0, 2, 2] = np.nan
                return
            if problem == "unknown_field":
                group = file.groups["state"]
                variable = group.create_variable(
                    "unrecognizedIce",
                    group.variables["hIceMean"].dimensions,
                    dtype="f8",
                )
                variable[:] = 0
                return
        with h5py.File(path, "r+") as file:
            del file["state/sigma12"]
    elif problem in {"dtype", "dimensions"}:
        _replace_state_variable(path, wrong_dtype=problem == "dtype")
    elif problem in {
        "negative_steps",
        "boolean_steps",
        "elapsed",
        "string_elapsed",
        "cursor",
        "exhausted_cursor",
    }:
        key, value = {
            "negative_steps": ("completed_steps", -1),
            "boolean_steps": ("completed_steps", True),
            "elapsed": ("elapsed_microseconds", 1),
            "string_elapsed": ("elapsed_microseconds", "1200000000"),
            "cursor": ("forcing_step", 0),
            "exhausted_cursor": ("completed_steps", 5),
        }[problem]

        def alter_clock(clock: dict[str, Any]) -> None:
            clock[key] = value
            if problem == "exhausted_cursor":
                clock["forcing_step"] = value
                clock["elapsed_microseconds"] = value * 600_000_000

        _edit_json_attribute(path, "clock", alter_clock)
    elif problem == "missing_forcing":
        with h5py.File(path, "r+") as file:
            del file["definition/forcing/series/uWind"]
    elif problem in {"partitions", "invalid_settings", "grid", "array_reference"}:

        def alter(manifest: dict[str, Any]) -> None:
            if problem == "partitions":
                manifest["execution"]["partitions"] = [2, 1]
            elif problem == "invalid_settings":
                manifest["settings"]["nEVPsteps"] = 0
            elif problem == "grid":
                manifest["grid"]["ny"] += 1
            else:
                manifest["forcing"]["series"]["uWind"]["array"] = "/missing/uWind"

        _edit_json_attribute(path, "experiment", alter)
    else:
        with h5netcdf.File(path, "r+") as file:
            if problem == "version":
                file.attrs["veris_restart_version"] = 2
            elif problem == "missing_metadata":
                del file.attrs["experiment"]
            elif problem == "backend":
                file.attrs["backend"] = "gpu"
            elif problem == "partial":
                file.groups["state"].variables["partial"][0] = 1
            elif problem == "mean":
                file.groups["state"].attrs["mean"] = 1
            elif problem == "duplicate_json":
                file.attrs["clock"] = (
                    '{"completed_steps":2,"completed_steps":1,'
                    '"elapsed_microseconds":1200000000,"forcing_step":2}'
                )
            elif problem == "malformed_json":
                file.attrs["clock"] = '{"completed_steps":'
            else:
                raise AssertionError(f"unhandled corruption {problem}")


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("missing_field", "missing|complete|sigma12"),
        ("unknown_field", "unknown|unexpected|unrecognized"),
        ("dtype", "dtype|precision"),
        ("dimensions", "dimension|shape"),
        ("nonfinite", "finite|hIceMean"),
        ("version", "version|schema"),
        ("missing_metadata", "metadata|experiment"),
        ("negative_steps", "step|clock"),
        ("boolean_steps", "integer|step|clock"),
        ("elapsed", "elapsed|time|clock"),
        ("string_elapsed", "integer|elapsed|time|clock"),
        ("cursor", "forcing|cursor|step"),
        ("exhausted_cursor", "forcing|length|exhaust|step"),
        ("partitions", "serial|partition|distributed"),
        ("backend", "backend"),
        ("invalid_settings", "nEVPsteps"),
        ("grid", "shape|grid|dimension"),
        ("array_reference", "array|reference|missing"),
        ("missing_forcing", "array|forcing|missing|reference"),
        ("partial", "partial|complete"),
        ("mean", "mean|instantaneous"),
        ("duplicate_json", "duplicate"),
        ("malformed_json", "JSON|clock|metadata"),
    ],
)
def test_corrupt_restart_fails_with_path_and_reason_before_device_allocation(
    restart_path: Path, problem: str, reason: str
) -> None:
    """No invalid checkpoint may silently default missing fields or select devices."""
    _corrupt(restart_path, problem)
    script = """
import re
import sys
import jax
from jax._src import xla_bridge
import veris
jax.config.update("jax_enable_x64", True)
assert not xla_bridge.backends_are_initialized()
try:
    veris.Simulation.from_restart(sys.argv[1])
except (ValueError, TypeError) as error:
    assert sys.argv[1] in str(error), str(error)
    assert re.search(sys.argv[2], str(error), re.IGNORECASE), str(error)
else:
    raise AssertionError("ERROR invalid restart accepted")
assert not xla_bridge.backends_are_initialized(), "ERROR restart allocated before validation"
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(restart_path), reason],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]


@pytest.mark.parametrize("record_kind", ["snapshot", "mean"])
def test_visualization_snapshot_and_mean_are_not_complete_restarts(
    tmp_path: Path, record_kind: str
) -> None:
    path = tmp_path / f"ordinary-{record_kind}.nc"
    with veris.Simulation.initialize(_experiment("growth", steps=2, seconds=1)) as sim:
        if record_kind == "snapshot":
            write_snapshot(
                path,
                sim.state,
                elapsed=sim.elapsed,
                conf=sim.configuration,
                phys=sim.physical_constants,
            )
        else:
            output = OutputSettings(
                streams=(
                    Stream(
                        "snapshot",
                        ("hIceMean",),
                        timedelta(seconds=1),
                        timedelta(seconds=2),
                    ),
                ),
            )
            with OutputManager(path, output) as manager:
                manager.sample(sim.state, sim.elapsed)
                sim.step()
                manager.sample(sim.state, sim.elapsed)
                sim.step()
                manager.sample(sim.state, sim.elapsed)
    with pytest.raises(ValueError, match="restart|snapshot|mean|kind|schema"):
        veris.Simulation.from_restart(path)


def test_restart_records_actual_backend_when_definition_honors_caller(
    tmp_path: Path,
) -> None:
    path = tmp_path / "implicit-backend.nc"
    definition = replace(_experiment("growth", steps=2), execution=veris.Execution())
    with veris.Simulation.initialize(definition) as original:
        original.step()
        original.save_restart(path)
        actual = next(iter(original.state.hIceMean.devices())).platform
        expected = original.state
    with h5netcdf.File(path, "r") as file:
        assert file.attrs["backend"] == actual
        assert json.loads(file.attrs["experiment"])["execution"]["backend"] is None
    with veris.Simulation.from_restart(path) as resumed:
        assert next(iter(resumed.state.hIceMean.devices())).platform == actual
        _assert_same_state(resumed.state, expected)


@pytest.mark.parametrize("operation", ["save", "restore"])
def test_restart_file_effects_are_rejected_inside_jax_transformations(
    tmp_path: Path, operation: str
) -> None:
    path = tmp_path / "transformed.nc"
    with veris.Simulation.initialize(_experiment("growth", steps=1)) as simulation:
        if operation == "restore":
            simulation.save_restart(path)
        before = path.read_bytes() if path.exists() else None

        def transformed() -> int:
            if operation == "save":
                simulation.save_restart(path)
            else:
                veris.Simulation.from_restart(path)
            return 0

        with pytest.raises(RuntimeError, match="outside|transform|host"):
            jax.jit(transformed)()
        assert (path.read_bytes() if path.exists() else None) == before


def test_distributed_restart_save_is_rejected_explicitly_before_file_creation(
    tmp_path: Path,
) -> None:
    """A two-device CPU mesh is unsupported even when all shards are addressable."""
    path = tmp_path / "distributed.nc"
    script = """
import sys
import jax
import numpy as np
from jax.sharding import Mesh
import veris
jax.config.update("jax_enable_x64", True)
mesh = Mesh(np.asarray(jax.devices("cpu")).reshape(2, 1), ("x", "y"))
definition = veris.Experiment.builtin(
    "dynamics", grid=veris.Grid(4, 2), steps=1,
    execution=veris.Execution(backend="cpu", partitions=(2, 1)),
)
with veris.Simulation.initialize(definition, mesh=mesh) as simulation:
    try:
        simulation.save_restart(sys.argv[1])
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("serial", "partition", "distributed"))
    else:
        raise AssertionError("ERROR distributed restart save accepted")
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    assert not path.exists()
