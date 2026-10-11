"""Selected input commits transactionally and preserves forcing/output position.

Ordinary storage snapshots deliberately
serve as selected inputs; complete restart has its separate stricter contract.
"""

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import veris
from veris.io import OutputSettings, Stream, read_record, write_snapshot


def _definition(*, dtype: str = "float32", output: OutputSettings | None = None) -> Any:
    return veris.Experiment.builtin(
        "growth",
        settings={"dtype": dtype, "deltatTherm": 1},
        forcing=veris.Forcing(series={"uWind": [3, 4, 5, 6]}),
        steps=4,
        output=output,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hIceMean", -1),
        ("hIceMean", np.nan),
        ("hIceMean", np.inf),
        ("hIceMean", 1e100),
        ("Area", -0.1),
        ("Area", 1.1),
        ("TSurf", 0),
        ("TSurf", 1e-100),
        ("wSpeed", -1),
        ("ocSalt", -1),
        ("theta", 0),
    ],
)
def test_invalid_selected_physical_fields_leave_state_time_and_forcing_usable(
    tmp_path: Path, field: str, value: float
) -> None:
    path = tmp_path / "invalid.nc"
    write_snapshot(path, {field: np.full((6, 6), value, dtype=np.float64)})
    with veris.Simulation.initialize(_definition()) as simulation:
        simulation.run(1)
        state, elapsed, cursor = (
            simulation.state,
            simulation.elapsed,
            simulation.completed_steps,
        )
        with pytest.raises(ValueError, match=field):
            simulation.load_fields(path, (field,))
        assert simulation.state is state and not simulation.closed
        assert simulation.elapsed == elapsed and simulation.completed_steps == cursor
        final = simulation.run(1)
        np.testing.assert_array_equal(final.state.uWind, 4)


def test_all_selected_fields_validate_before_any_valid_field_commits(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.nc"
    write_snapshot(
        path, {"hIceMean": np.full((6, 6), 2.0), "Area": np.full((6, 6), 1.1)}
    )
    with veris.Simulation.initialize(_definition()) as simulation:
        before = simulation.state
        with pytest.raises(ValueError, match="Area"):
            simulation.load_fields(path, ("hIceMean", "Area"))
        assert simulation.state is before


def test_selected_input_casts_representable_float64_into_model_float32(
    tmp_path: Path,
) -> None:
    path = tmp_path / "precision.nc"
    write_snapshot(path, {"hIceMean": np.full((6, 6), 1.25, dtype=np.float64)})
    with veris.Simulation.initialize(_definition()) as simulation:
        simulation.load_fields(path, ("hIceMean",))
        assert simulation.state.hIceMean.dtype == np.dtype("float32")
        np.testing.assert_array_equal(simulation.state.hIceMean, 1.25)
        assert simulation.completed_steps == 0 and simulation.elapsed == timedelta(0)


def test_selected_load_preserves_pending_means_and_next_prescribed_forcing(
    tmp_path: Path,
) -> None:
    output = OutputSettings(
        streams=(
            Stream("mean", ("uWind",), timedelta(seconds=1), timedelta(seconds=4)),
        )
    )
    history, input_path = tmp_path / "history.nc", tmp_path / "wind.nc"
    write_snapshot(input_path, {"uWind": np.full((6, 6), 99.0)})
    with veris.Simulation.initialize(
        _definition(dtype="float64", output=output), output_path=history
    ) as simulation:
        simulation.run(1)
        simulation.load_fields(input_path, ("uWind",))
        np.testing.assert_array_equal(simulation.state.wSpeed, 99)
        assert simulation.completed_steps == 1 and simulation.elapsed == timedelta(
            seconds=1
        )
        final = simulation.run(3)
        np.testing.assert_array_equal(final.state.uWind, 6)
    record = read_record(history, stream="mean")
    assert record.bounds == (0, 4) and record.count == 4
    # Pre-transition samples are initial3, loaded99, prescribed4, prescribed5.
    np.testing.assert_array_equal(record.fields["uWind"], 27.75)


@pytest.mark.parametrize("cyclic_y", [True, False])
def test_selected_coastal_ice_and_velocities_preserve_cell_means_and_blocked_faces(
    tmp_path: Path, cyclic_y: bool
) -> None:
    """A fractional ocean mask marks wet cells; selected values are not rescaled."""
    nx, ny = 4, 5
    surface = np.ones((nx, ny))
    surface[1, 2], surface[2, 3] = 0, 0.5
    west, south = (
        surface * np.roll(surface, 1, axis=0),
        surface * np.roll(surface, 1, axis=1),
    )
    geometry = veris.OceanGeometry(
        maskT=surface[:, :, None],
        maskU=west[:, :, None],
        maskV=south[:, :, None],
        ht=-80 * np.ones((nx, ny)),
        coriolis_t=np.zeros((nx, ny)),
        dxt=np.full(nx, 2000),
        dxu=np.full(nx, 2000),
        dyt=np.full(ny, 2000),
        dyu=np.full(ny, 2000),
        area_t=np.full((nx, ny), 4e6),
        area_u=np.full((nx, ny), 4e6),
        area_v=np.full((nx, ny), 4e6),
    )
    definition = veris.Experiment(
        grid=veris.Grid(nx, ny),
        geometry=geometry,
        settings={"dtype": "float64", "enable_cyclic_y": cyclic_y},
        initial_conditions={"hIceMean": 1, "Area": 0.5},
        steps=0,
    )
    path = tmp_path / "coast.nc"
    values = {
        "hIceMean": 2.0,
        "hSnowMean": 0.1,
        "Area": 0.75,
        "uIce": 0.03,
        "vIce": -0.04,
    }
    write_snapshot(
        path, {name: np.full((nx + 4, ny + 4), value) for name, value in values.items()}
    )

    def expected(physical: np.ndarray, normal: bool = False) -> np.ndarray:
        x = (np.arange(nx + 4) - 2) % nx
        y = np.arange(ny + 4) - 2
        y = y % ny if cyclic_y else np.clip(y, 0, ny - 1)
        array = physical[np.ix_(x, y)].copy()
        if not cyclic_y:
            array[:, :2] = array[:, -2:] = 0
            if normal:
                array[:, 2] = 0
        return array

    with veris.Simulation.initialize(definition) as simulation:
        original_forcing = simulation.experiment.forcing
        simulation.load_fields(path, tuple(values))
        for name, mask in (
            ("hIceMean", surface),
            ("hSnowMean", surface),
            ("Area", surface),
            ("uIce", west),
            ("vIce", south),
        ):
            np.testing.assert_array_equal(
                getattr(simulation.state, name),
                expected(np.where(mask != 0, values[name], 0), normal=name == "vIce"),
            )
        assert simulation.state.hIceMean[4, 5] == 2
        assert simulation.state.Area[4, 5] == 0.75
        assert simulation.experiment.forcing is original_forcing
        assert simulation.elapsed == timedelta(0) and simulation.completed_steps == 0
