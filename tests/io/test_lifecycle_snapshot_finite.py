"""Snapshots reject selected physical nonfinite data before creating files."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

import veris
from veris.io.lifecycle import snapshot


def test_selected_snapshot_nonfinite_rejects_without_file(tmp_path: Path) -> None:
    definition = veris.Experiment.builtin("growth", steps=0)
    state, conf, phys = veris.initialize_experiment(definition)
    invalid = replace(state, hIceMean=state.hIceMean.at[2, 2].set(float("nan")))
    destination = tmp_path / "invalid.nc"
    with pytest.raises(ValueError, match="nonfinite output in hIceMean"):
        snapshot(
            destination,
            invalid,
            definition,
            conf,
            phys,
            timedelta(0),
            None,
            ("hIceMean",),
        )
    assert not destination.exists()


def test_snapshot_only_validates_selected_physical_cells(tmp_path: Path) -> None:
    definition = veris.Experiment.builtin("growth", steps=0)
    state, conf, phys = veris.initialize_experiment(definition)
    invalid = replace(
        state,
        hIceMean=state.hIceMean.at[0, 0].set(float("nan")),
        uIce=state.uIce.at[2, 2].set(float("nan")),
    )
    destination = tmp_path / "valid.nc"
    snapshot(
        destination, invalid, definition, conf, phys, timedelta(0), None, ("hIceMean",)
    )
    assert destination.exists()
