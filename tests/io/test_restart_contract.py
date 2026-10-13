"""Exact checkpoint clocks, halos, publication and resumed-output invariants."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import numpy as np
import pytest

import veris
from veris.io import OutputSettings, Stream, read_record


def test_restart_at_zero_does_not_reemit_checkpoint_instant(tmp_path: Path) -> None:
    """A restored writer starts strictly after even a zero-step checkpoint."""
    checkpoint = tmp_path / "zero-restart.nc"
    original_path, resumed_path = tmp_path / "original.nc", tmp_path / "resumed.nc"
    output = OutputSettings(
        streams=(
            Stream("instant", ("uWind",), timedelta(seconds=1)),
            Stream("mean", ("uWind",), timedelta(seconds=1), timedelta(seconds=2)),
        ),
        sample_initial=True,
    )
    definition = veris.Experiment.builtin(
        "growth",
        steps=2,
        settings={"dtype": "float32", "deltatTherm": 1},
        output=output,
    )
    with veris.Simulation.initialize(definition, output_path=original_path) as original:
        original.run(0)
        assert read_record(original_path, stream="instant").time == 0
        expected_mean = np.asarray(original.state.uWind)[2:-2, 2:-2].copy()
        original.save_restart(checkpoint)
    with veris.Simulation.from_restart(checkpoint, output_path=resumed_path) as resumed:
        resumed.run(0)
        assert not resumed_path.exists(), "ERROR restart repeated checkpoint instant"
        resumed.run(2)
    with h5netcdf.File(resumed_path, "r") as file:
        np.testing.assert_array_equal(
            file.groups["instant"].variables["time"][:], [1, 2]
        )
        mean = file.groups["mean"]
        np.testing.assert_array_equal(mean.variables["time_bounds"][:], [[0, 2]])
        np.testing.assert_array_equal(mean.variables["sample_count"][:], [2])
        np.testing.assert_array_equal(mean.variables["uWind"][0], expected_mean)


def _growth(**options: Any) -> Any:
    """Small real physics with explicit CPU execution and one-second clock."""
    return veris.Experiment.builtin(
        "growth",
        execution=veris.Execution(backend="cpu"),
        settings={"dtype": "float64", "deltatTherm": 1},
        **options,
    )


def test_explicit_completed_count_can_exceed_default_target(tmp_path: Path) -> None:
    """Experiment.steps is a default target, not an absolute checkpoint bound."""
    path = tmp_path / "extended.nc"
    with veris.Simulation.initialize(_growth(steps=1)) as original:
        original.run(3)
        original.save_restart(path)
    with veris.Simulation.from_restart(path) as resumed:
        assert resumed.completed_steps == 3
        assert resumed.run().steps == 0
        assert resumed.step().completed_steps == 4


def test_no_output_restart_accepts_exact_clock_beyond_int64_and_gregorian_range(
    tmp_path: Path,
) -> None:
    """No-output clocks use timedelta bounds and Python integer counts only."""
    path = tmp_path / "long-clock.nc"
    with veris.Simulation.initialize(_growth(steps=0)) as original:
        original.save_restart(path)
    completed = 20_000_000_000_000
    microseconds = completed * 1_000_000
    with h5netcdf.File(path, "r+") as file:
        file.attrs["clock"] = json.dumps(
            {
                "completed_steps": completed,
                "forcing_step": completed,
                "elapsed_microseconds": microseconds,
            }
        )
        file.groups["state"].variables["time"][0] = microseconds / 1_000_000
        file.groups["state"].variables["time_bounds"][0] = [completed, completed]
    with veris.Simulation.from_restart(path) as resumed:
        assert resumed.completed_steps == completed
        assert resumed.elapsed == timedelta(microseconds=microseconds)
        assert resumed.run(0).elapsed == timedelta(microseconds=microseconds)


def test_restart_codec_never_sends_arrays_to_json_list_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete series arrays go directly through the private codec callback."""
    import veris.model.experiment_file as codec

    definition = replace(
        _growth(steps=2),
        forcing=veris.Forcing(
            series={
                "uWind": np.arange(2.0),
                "theta": np.full(2, 270.0),
            }
        ),
    )
    original_plain = codec._plain

    def scalar_only(value: Any) -> Any:
        assert not isinstance(value, np.ndarray), "ERROR restart made a JSON array copy"
        return original_plain(value)

    monkeypatch.setattr(codec, "_plain", scalar_only)
    with veris.Simulation.initialize(definition) as original:
        original.save_restart(tmp_path / "arrays.nc")


def test_publication_failure_cleans_temporary_file_and_preserves_live_simulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atomic publish errors leave neither a partial checkpoint nor leaked temp."""
    from veris.io import restart

    path = tmp_path / "failed.nc"

    def failed_link(*args: Any, **kwargs: Any) -> None:
        raise OSError("publication failed")

    monkeypatch.setattr(restart.os, "link", failed_link)
    with veris.Simulation.initialize(_growth(steps=2)) as original:
        state = original.state
        with pytest.raises(OSError, match="publication failed"):
            original.save_restart(path)
        assert original.state is state
        assert original.completed_steps == 0 and not original.closed
        assert list(tmp_path.iterdir()) == []


def test_original_version_one_execution_schema_remains_readable(tmp_path: Path) -> None:
    destination = tmp_path / "original-v1.nc"
    definition = veris.Experiment.builtin("growth", steps=0)
    with veris.Simulation.initialize(definition) as model:
        model.save_restart(destination)
    with h5netcdf.File(destination, "r+") as file:
        encoded = json.loads(file.attrs["experiment"])
        encoded["execution"].pop("all_devices")
        encoded["execution"].pop("distributed")
        file.attrs["experiment"] = json.dumps(encoded)
    with veris.Simulation.from_restart(destination) as model:
        assert model.completed_steps == 0
        assert model.experiment.execution.distributed is None
        assert not model.experiment.execution.all_devices


@pytest.mark.parametrize(
    "controls",
    [
        {"all_devices": True},
        {
            "distributed": {
                "coordinator_address": "host:12345",
                "num_processes": 2,
                "process_id": 0,
            }
        },
    ],
)
def test_restart_rejects_unresolved_launch_controls_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, controls: dict[str, Any]
) -> None:
    import jax

    destination = tmp_path / "launch-controls.nc"
    with veris.Simulation.initialize(
        veris.Experiment.builtin("growth", steps=0)
    ) as model:
        model.save_restart(destination)
    with h5netcdf.File(destination, "r+") as file:
        encoded = json.loads(file.attrs["experiment"])
        encoded["execution"].update(controls, backend=str(file.attrs["backend"]))
        file.attrs["experiment"] = json.dumps(encoded)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR invalid restart launch controls selected devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    with pytest.raises(ValueError, match="serial|distributed"):
        veris.Simulation.from_restart(destination)
