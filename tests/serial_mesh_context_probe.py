"""Real CPU serial lifecycle ownership inside a caller's ambient device mesh."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jax
import numpy as np

import veris
from veris.io import OutputSettings, Stream, read_record


def _single_device(tree: Any, device: Any, label: str) -> None:
    for array in jax.tree.leaves(tree):
        assert array.devices() == {device}, (
            f"ERROR {label} acquired ambient mesh devices"
        )


def _compare(left: Any, right: Any, label: str) -> None:
    for actual, expected in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape and a.dtype == b.dtype
        error = np.abs(a - b)
        assert np.isfinite(a).all() and np.all(error <= 2e-11 * (1 + np.abs(b))), (
            f"ERROR {label}: max_abs={error.max():.6g}"
        )


def _definition(backend: str | None, output: OutputSettings) -> veris.Experiment:
    x, y = np.indices((4, 6))
    return veris.Experiment.builtin(
        "growth",
        grid=veris.Grid(4, 6),
        steps=2,
        execution=veris.Execution(backend=backend),
        settings={"deltatTherm": 600},
        initial_conditions={"hIceMean": 1 + 0.01 * x + 0.02 * y, "Area": 0.8},
        forcing=veris.Forcing(
            constant={"vWind": 1.0, "Qsw": 0.0},
            series={
                "uWind": np.stack((2 + 0.1 * x + 0.03 * y, 3 + 0.04 * x + 0.02 * y)),
                "Qnet": [80.0, 100.0],
            },
        ),
        output=output,
    )


def _observe(state: Any, diagnostics: Any) -> tuple[Any, Any]:
    return state.hIceMean[2:-2, 2:-2], diagnostics.EmPmR[2:-2, 2:-2]


def _check(inside: bool, backend: str | None, mesh: Any, directory: Path) -> None:
    devices = jax.devices("cpu")
    chosen = devices[-1] if backend is None else devices[0]
    label = f"inside{inside}-backend{backend}"
    output = OutputSettings(
        streams=(
            Stream("instant", ("hIceMean",), timedelta(seconds=600)),
            Stream(
                "mean", ("hIceMean",), timedelta(seconds=600), timedelta(seconds=1200)
            ),
        )
    )
    definition = _definition(backend, output)
    history = directory / f"{label}-history.nc"
    snapshot = directory / f"{label}-snapshot.nc"
    checkpoint = directory / f"{label}-restart.nc"
    with jax.default_device(devices[-1]):
        with veris.Simulation.initialize(
            replace(definition, output=replace(output, enabled=False))
        ) as reference:
            initial_ice = np.asarray(reference.state.hIceMean)[2:-2, 2:-2].copy()
            expected = reference.run(2, observe=_observe)
        if inside:
            with jax.set_mesh(mesh):
                initialized, _, _ = veris.initialize_experiment(definition)
                _single_device(initialized, chosen, label + " public initializer")
                simulation = veris.Simulation.initialize(
                    definition, output_path=history
                )
                assert jax.sharding.get_abstract_mesh() == mesh.abstract_mesh
        else:
            initialized, _, _ = veris.initialize_experiment(definition)
            _single_device(initialized, chosen, label + " public initializer")
            simulation = veris.Simulation.initialize(definition, output_path=history)
        with simulation, jax.set_mesh(mesh):
            _single_device(simulation.state, chosen, label + " initialized State")
            first = simulation.run(1, observe=_observe)
            _single_device(
                (first.state, first.diagnostics, first.observations),
                chosen,
                label + " first run",
            )
            assert jax.sharding.get_abstract_mesh() == mesh.abstract_mesh
            simulation.save_restart(checkpoint)
            with veris.Simulation.from_restart(
                checkpoint, output_settings=replace(output, enabled=False)
            ) as restored:
                assert len(restored.state.Area.devices()) == 1
                resumed = restored.run(1)
                assert len(resumed.state.Area.devices()) == 1
                _compare(resumed.state, expected.state, label + " restored run")
            assert jax.sharding.get_abstract_mesh() == mesh.abstract_mesh
            second = simulation.run(1, observe=_observe)
            _single_device(
                (second.state, second.diagnostics, second.observations),
                chosen,
                label + " second run",
            )
            _compare(
                (second.state, second.diagnostics),
                (expected.state, expected.diagnostics),
                label + " final",
            )
            _compare(
                first.observations,
                jax.tree.map(lambda a: a[:1], expected.observations),
                label + " first observation",
            )
            _compare(
                second.observations,
                jax.tree.map(lambda a: a[1:], expected.observations),
                label + " second observation",
            )
            simulation.snapshot(snapshot, variables=("hIceMean",))
            assert jax.sharding.get_abstract_mesh() == mesh.abstract_mesh
        saved = read_record(snapshot)
        assert saved.time == 1200
        _compare(
            saved.fields["hIceMean"],
            expected.state.hIceMean[2:-2, 2:-2],
            label + " snapshot",
        )
        instant = read_record(history, stream="instant")
        mean = read_record(history, stream="mean")
        assert instant.time == 1200 and instant.bounds == (1200, 1200)
        assert mean.time == 600 and mean.bounds == (0, 1200) and mean.mean
        assert mean.count == 2 and not mean.partial
        assert mean.fields["hIceMean"].dtype == np.dtype("float64")
        _compare(
            instant.fields["hIceMean"], saved.fields["hIceMean"], label + " instant"
        )
        _compare(
            mean.fields["hIceMean"],
            (initial_ice + np.asarray(expected.observations[0])[0]) / 2,
            label + " mean",
        )
        assert jax.sharding.get_abstract_mesh().empty


def main() -> None:
    """Exercise two/four devices, both initialization contexts and backend choices."""
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices("cpu")
    assert len(devices) in (2, 4), "ERROR expected two or four real CPU devices"
    mesh = jax.make_mesh((1, len(devices)), ("x", "y"), devices=devices)
    with TemporaryDirectory() as location:
        for inside in (False, True):
            for backend in (None, "cpu"):
                _check(inside, backend, mesh, Path(location))
    print(
        f"serial ambient mesh: {len(devices)} CPU devices, four lifecycle contexts passed"
    )


if __name__ == "__main__":
    main()
