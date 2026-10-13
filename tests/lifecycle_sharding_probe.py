"""Draft real four-CPU public initialization/run/output checks for Task 7.

Copy into tests before wiring the launcher. This script was not executed during
draft preparation. Initial all-field packing is an independent global-index
oracle, including stored closed-y wall columns; output uses public snapshots.
"""

from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import h5netcdf
import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import veris
from veris.execution import simulation
from veris.io import OutputSettings, Stream, read_record


def _pack(field: Any, partitions: tuple[int, int], cyclic_y: bool) -> np.ndarray:
    """Index serial physical cells and exterior wall columns without halo helpers."""
    source = np.asarray(field)
    gx, gy = source.shape[0] - 4, source.shape[1] - 4
    px, py = partitions
    nx, ny = gx // px, gy // py
    packed = np.empty((px * (nx + 4), py * (ny + 4)), dtype=source.dtype)
    for rx in range(px):
        for ry in range(py):
            x = (rx * nx + np.arange(nx + 4) - 2) % gx + 2
            global_y = ry * ny + np.arange(ny + 4) - 2
            y = global_y % gy + 2 if cyclic_y else global_y + 2
            packed[
                rx * (nx + 4) : (rx + 1) * (nx + 4),
                ry * (ny + 4) : (ry + 1) * (ny + 4),
            ] = source[np.ix_(x, y)]
    return packed


def _compare(actual: Any, expected: Any, label: str, tolerance: float = 0) -> None:
    left, right = np.asarray(actual), np.asarray(expected)
    assert left.shape == right.shape and left.dtype == right.dtype, (
        f"ERROR {label}: shape/dtype"
    )
    assert np.isfinite(left).all() and np.isfinite(right).all(), (
        f"ERROR {label}: nonfinite"
    )
    error = np.abs(left - right)
    assert np.all(error <= tolerance * (1 + np.abs(right))), (
        f"ERROR {label}: max_abs={error.max():.6g}"
    )


def _custom(grid: Any, cyclic_y: bool, output: OutputSettings) -> Any:
    x, y = np.indices((grid.nx, grid.ny))
    wet = np.ones((grid.nx, grid.ny))
    wet[4:6, 6:8] = 0
    wet[7, 10] = 0.5
    dx, dy = 8000 + 10 * np.arange(grid.nx), 9000 + 20 * np.arange(grid.ny)
    area = dx[:, None] * dy[None, :]
    geometry = veris.OceanGeometry(
        maskT=wet[..., None],
        maskU=(wet * np.roll(wet, 1, 0))[..., None],
        maskV=(wet * np.roll(wet, 1, 1))[..., None],
        ht=-(100 + x + y),
        coriolis_t=np.full(wet.shape, 1e-4),
        dxt=dx,
        dxu=dx + 5,
        dyt=dy,
        dyu=dy + 10,
        area_t=area,
        area_u=area + 100,
        area_v=area + 200,
    )
    return veris.Experiment(
        grid=grid,
        geometry=geometry,
        settings={
            "enable_cyclic_y": cyclic_y,
            "deltatDyn": 600.0,
            "deltatTherm": 600.0,
            "nEVPsteps": 2,
        },
        initial_conditions={
            "hIceMean": 1 + 0.01 * x + 0.02 * y,
            "hSnowMean": 0.05,
            "Area": 0.8,
            "TSurf": 260,
        },
        forcing=veris.Forcing(
            constant={
                "vWind": 1.0,
                "theta": 271.35,
                "ocSalt": 34.7,
                "ATemp": 260,
                "LWdown": 260,
                "aqh": 0.001,
                "Qsw": 0,
            },
            series={
                "uWind": np.stack((3 + 0.01 * x + 0.02 * y, 4 + 0.02 * x + 0.01 * y)),
                "Qnet": [80, 110],
            },
        ),
        steps=2,
        output=output,
    )


def _check(
    name: str,
    layout: tuple[int, int],
    cyclic_y: bool,
    directory: Path,
    serial_references: dict[tuple[str, bool], tuple[Any, Any, Path]],
) -> None:
    grid = veris.Grid(12, 16)
    output = OutputSettings(
        streams=(
            Stream(
                "mean",
                ("hIceMean", "uIce"),
                timedelta(seconds=600),
                timedelta(seconds=1200),
            ),
            Stream("instant", ("hIceMean", "uIce"), timedelta(seconds=600)),
        )
    )
    definition = (
        _custom(grid, cyclic_y, output)
        if name == "custom"
        else veris.Experiment.builtin(
            name,
            grid=grid,
            settings={
                "enable_cyclic_y": cyclic_y,
                "deltatDyn": 600.0,
                "deltatTherm": 600.0,
                "nEVPsteps": 2,
            },
            steps=2,
            output=output,
        )
    )
    mesh = jax.make_mesh(layout, ("x", "y"), devices=jax.devices("cpu"))
    parallel_definition = replace(
        definition, execution=veris.Execution(backend="cpu", partitions=layout)
    )
    label = f"{name}-{layout[0]}x{layout[1]}-cyclic{cyclic_y}"
    key = name, cyclic_y
    if key not in serial_references:
        serial_path = directory / f"{name}-cyclic{cyclic_y}-serial.nc"
        # Initialization/execution still exercise serial ownership inside the
        # first caller's ambient mesh. Later layouts reuse the same physical
        # oracle while retaining their own real communication/output work.
        with (
            jax.set_mesh(mesh),
            veris.Simulation.initialize(definition, output_path=serial_path) as serial,
        ):
            initial = serial.state
            expected = serial.run(2)
        serial_references[key] = initial, expected, serial_path
    initial, expected, serial_path = serial_references[key]
    parallel_path = directory / f"{label}-parallel.nc"
    final_path = directory / f"{label}-final.nc"
    with (
        jax.set_mesh(mesh),
        veris.Simulation.initialize(
            parallel_definition, output_path=parallel_path, mesh=mesh
        ) as parallel,
    ):
        assert (
            parallel.configuration.nx,
            parallel.configuration.ny,
        ) == grid.local_shape(layout)
        for item in fields(initial):
            _compare(
                getattr(parallel.state, item.name),
                _pack(getattr(initial, item.name), layout, cyclic_y),
                f"{label} initial {item.name}",
            )
        parallel.run(1)
        parallel.run(1)
        parallel.snapshot(final_path)
    saved = read_record(final_path)
    for item in fields(expected.state):
        _compare(
            saved.fields[item.name],
            getattr(expected.state, item.name)[2:-2, 2:-2],
            f"{label} final {item.name}",
            1e-10,
        )
    with (
        h5netcdf.File(serial_path, "r") as left,
        h5netcdf.File(parallel_path, "r") as right,
    ):
        assert set(left.groups) == set(right.groups)
        for group_name in left.groups:
            for variable_name, reference in left.groups[group_name].variables.items():
                _compare(
                    right.groups[group_name].variables[variable_name][:],
                    reference[:],
                    f"{label} output {group_name}/{variable_name}",
                    1e-10,
                )


def _check_small_initialization(
    name: str, layout: tuple[int, int], cyclic_y: bool
) -> None:
    """Require exact full storage even when a local interior has only two cells."""
    px, py = layout
    grid = veris.Grid(max(4, 2 * px), max(4, 2 * py))
    settings = {
        "enable_cyclic_y": cyclic_y,
        "deltatDyn": 600.0,
        "deltatTherm": 600.0,
        "nEVPsteps": 2,
    }
    if name == "custom":
        x, y = np.indices((grid.nx, grid.ny))
        mask = np.ones((grid.nx, grid.ny))
        mask[1:3, 1:3] = 0
        dx, dy = 8000 + 10 * np.arange(grid.nx), 9000 + 20 * np.arange(grid.ny)
        area = dx[:, None] * dy[None, :]
        definition = veris.Experiment(
            grid=grid,
            settings=settings,
            geometry=veris.OceanGeometry(
                maskT=mask[..., None],
                maskU=(mask * np.roll(mask, 1, 0))[..., None],
                maskV=(mask * np.roll(mask, 1, 1))[..., None],
                ht=-(100 + x + y),
                coriolis_t=np.full(mask.shape, 1e-4),
                dxt=dx,
                dxu=dx + 5,
                dyt=dy,
                dyu=dy + 10,
                area_t=area,
                area_u=area + 100,
                area_v=area + 200,
            ),
            initial_conditions={"hIceMean": 1 + 0.01 * x + 0.02 * y, "Area": 0.8},
            forcing=veris.Forcing(constant={"uWind": x + 2 * y, "vWind": 1.0}),
        )
    else:
        definition = veris.Experiment.builtin(name, grid=grid, settings=settings)
    serial, _, _ = veris.initialize_experiment(definition)
    mesh = jax.make_mesh(layout, ("x", "y"), devices=jax.devices("cpu"))
    parallel, _, _ = veris.initialize_experiment(
        replace(
            definition, execution=veris.Execution(backend="cpu", partitions=layout)
        ),
        mesh=mesh,
    )
    for item in fields(serial):
        _compare(
            getattr(parallel, item.name),
            _pack(getattr(serial, item.name), layout, cyclic_y),
            f"small {name}/{layout}/{cyclic_y} {item.name}",
        )


def main() -> None:
    """Run each communication orientation once per preset/boundary policy."""
    jax.config.update("jax_enable_x64", True)
    assert len(jax.devices("cpu")) == 4, "ERROR expected four CPU devices"
    original = simulation.run_timed
    spatial_calls = []

    def inspect_inputs(state: Any, *args: Any, **kwargs: Any) -> Any:
        """Inspect dynamic placement while retaining the real scan and physics."""
        inputs = kwargs.get("inputs")
        state_sharding = state[0].Area.sharding
        if (
            inputs
            and isinstance(state_sharding, NamedSharding)
            and state_sharding.spec == P("x", "y")
        ):
            for name, value in inputs.items():
                expected = P(None, "x", "y") if value.ndim == 3 else P()
                assert (
                    isinstance(value.sharding, NamedSharding)
                    and value.sharding.spec == expected
                ), f"ERROR {name} forcing series placement"
                if value.ndim == 3:
                    spatial_calls.append(name)
        return original(state, *args, **kwargs)

    with (
        patch.object(simulation, "run_timed", new=inspect_inputs),
        TemporaryDirectory() as location,
    ):
        directory = Path(location)
        serial_references: dict[tuple[str, bool], tuple[Any, Any, Path]] = {}
        for layout in ((1, 4), (4, 1), (2, 2)):
            for name in ("island", "dynamics", "growth", "custom"):
                for cyclic_y in (False, True):
                    _check(name, layout, cyclic_y, directory, serial_references)
    for layout in ((1, 4), (4, 1), (2, 2)):
        for name in ("island", "dynamics", "growth", "custom"):
            for cyclic_y in (False, True):
                _check_small_initialization(name, layout, cyclic_y)
    assert len(spatial_calls) == 12, (
        "ERROR missing split-run spatial placement evidence"
    )
    print(
        "public lifecycle: 24 four-CPU preset/custom initialization, run and output cases passed"
    )


if __name__ == "__main__":
    main()
