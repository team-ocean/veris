"""Finite snapshot failure reaches writer and nonwriter resource participants."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils

import veris
from veris.io import lifecycle as simulation_io
from veris.io.storage import storage_fields


@pytest.mark.parametrize("rank", [0, 1], ids=["writer", "nonwriter"])
def test_snapshot_nonfinite_failure_reaches_all_participants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    """Mock only cross-host resources; allocation/trim/finite checks stay real."""
    definition = veris.Experiment.builtin("growth", steps=0)
    state, conf, phys = veris.initialize_experiment(definition)
    invalid = replace(state, hIceMean=state.hIceMean.at[2, 2].set(np.nan))
    mesh = jax.make_mesh((1, 1), ("x", "y"), devices=jax.devices("cpu")[:1])
    events = []

    def collect(source: Any, names: tuple[str, ...]) -> Any:
        events.append("collect")
        assert names == ("hIceMean", "Area")
        return storage_fields(source, names) if rank == 0 else None

    def status(value: Any, is_source: bool | None = None) -> Any:
        events.append("status")
        assert is_source is (rank == 0)
        assert np.asarray(value).shape == ()
        if rank == 0:
            assert int(np.asarray(value)) == 0
        # Writer communicates the selected-field position, even when this
        # participant's collector returned None instead of physical arrays.
        return np.asarray(0, dtype=np.int32)

    monkeypatch.setattr(simulation_io, "distributed_collector", lambda _: collect)
    monkeypatch.setattr(jax, "process_index", lambda: rank)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", status)
    destination = tmp_path / f"invalid-rank{rank}.nc"
    with pytest.raises(ValueError, match="nonfinite output in hIceMean"):
        simulation_io.snapshot(
            destination,
            invalid,
            definition,
            conf,
            phys,
            timedelta(0),
            mesh,
            ("hIceMean", "Area"),
        )
    assert events == ["collect", "status"]
    assert not destination.exists()


@pytest.mark.parametrize("rank", [0, 1], ids=["writer", "nonwriter"])
def test_snapshot_collective_success_only_checks_selected_physical_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    """Successful finite preflight preserves writer ownership and selection."""
    definition = veris.Experiment.builtin("growth", steps=0)
    state, conf, phys = veris.initialize_experiment(definition)
    source = replace(
        state,
        hIceMean=state.hIceMean.at[0, 0].set(np.nan),
        uIce=state.uIce.at[2, 2].set(np.nan),
    )
    mesh = jax.make_mesh((1, 1), ("x", "y"), devices=jax.devices("cpu")[:1])
    events = []

    def collect(state: Any, names: tuple[str, ...]) -> Any:
        events.append("collect")
        assert names == ("hIceMean",)
        return storage_fields(state, names) if rank == 0 else None

    def status(value: Any, is_source: bool | None = None) -> Any:
        events.append("status")
        assert is_source is (rank == 0)
        assert int(np.asarray(value)) == -1
        return np.asarray(-1, dtype=np.int32)

    monkeypatch.setattr(simulation_io, "distributed_collector", lambda _: collect)
    monkeypatch.setattr(jax, "process_index", lambda: rank)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all", status)
    destination = tmp_path / f"valid-rank{rank}.nc"
    simulation_io.snapshot(
        destination, source, definition, conf, phys, timedelta(0), mesh, ("hIceMean",)
    )
    assert events == ["collect", "status"]
    assert destination.exists() is (rank == 0)
