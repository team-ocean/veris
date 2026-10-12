"""CLI-owned nested destinations retain legacy parent-creation behavior."""

from pathlib import Path

import numpy as np
import pytest

import veris
from veris.io import OutputSettings, read_record
from veris.setups import run_growth


def test_growth_cli_creates_separate_nested_final_and_history_parents(
    tmp_path: Path,
) -> None:
    """Zero-step real files retain selected variables and non-Gregorian origin."""
    final = tmp_path / "snapshots" / "nested" / "final.nc"
    history = tmp_path / "histories" / "nested" / "history.nc"
    assert not final.parent.exists() and not history.parent.exists()
    assert (
        run_growth.main(
            [
                "--steps",
                "0",
                "--output",
                str(final),
                "--netcdf",
                str(history),
                "--calendar",
                "360_day",
                "--start-date",
                "2000-02-30",
                "--io-variables",
                "hIceMean,Area",
            ]
        )
        is None
    )
    state, _, _ = run_growth.initialize()
    for path, stream in ((final, "snapshot"), (history, "instantaneous")):
        record = read_record(path, stream=stream)
        assert set(record.fields) == {"hIceMean", "Area"}
        assert record.calendar == "360_day" and record.time == 0
        assert record.units == "seconds since 2000-02-30 00:00:00"
        for name, actual in record.fields.items():
            np.testing.assert_array_equal(
                actual, np.asarray(getattr(state, name))[2:-2, 2:-2]
            )


def test_public_simulation_still_rejects_missing_output_parents(tmp_path: Path) -> None:
    """Directory creation belongs to the CLI's requested destinations only."""
    destination = tmp_path / "missing" / "output.nc"
    definition = veris.Experiment.builtin("growth", steps=0, output=OutputSettings())
    with pytest.raises(FileNotFoundError, match="parent directory"):
        veris.Simulation.initialize(definition, output_path=destination)
    with (
        veris.Simulation.initialize(
            veris.Experiment.builtin("growth", steps=0)
        ) as simulation,
        pytest.raises(FileNotFoundError, match="parent directory"),
    ):
        simulation.snapshot(destination)
    assert not destination.parent.exists()
