"""Requested post-step histories coexist with pre-step scheduled reductions.

Algebraic prefix-sum oracles check
timing/chunks; real growth checks State/Diagnostics observations through the
public facade. No pytest or numerical probes ran during preparation.
"""

from dataclasses import replace
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import veris
from veris.execution.rollout_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream


def _pair() -> tuple[dict[str, jax.Array], jax.Array]:
    return (
        {
            "Area": jnp.zeros((6, 7), dtype=jnp.float32),
            "hIceMean": jnp.ones((6, 7), dtype=jnp.float32),
        },
        jnp.array(0, dtype=jnp.float32),
    )


def _advance(carry: Any, increment: jax.Array) -> Any:
    state, _ = carry
    updated = {**state, "Area": state["Area"] + increment}
    return updated, updated["Area"][0, 0] + increment


def _select_history(carry: Any) -> dict[str, jax.Array]:
    return {"area": carry[0]["Area"][2, 2], "diagnostic": carry[1]}


def _streams() -> OutputSettings:
    return OutputSettings(
        streams=(
            Stream("instant", ("Area",), timedelta(seconds=2)),
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
        )
    )


def test_observed_scheduled_trajectory_executes_each_input_once_and_keeps_sample_roles(
    tmp_path: Path,
) -> None:
    executed, chunks = [], []

    def advance(carry: Any, increment: jax.Array) -> Any:
        jax.debug.callback(
            lambda value: executed.append(int(value)), increment, ordered=True
        )
        return _advance(carry, increment)

    path = tmp_path / "both.nc"
    with OutputManager(path, _streams()) as manager:
        observer, _ = output_callbacks(manager, 1)
        final, _, _ = run_timed(
            _pair(),
            advance,
            13,
            inputs=jnp.arange(1, 14, dtype=jnp.float32),
            observe=observer,
            select=lambda pair: pair[0],
            history_select=_select_history,
            history_sink=chunks.append,
        )
    jax.effects_barrier()
    assert executed == list(range(1, 14)), "ERROR observations replayed physics"
    history = jax.tree.map(lambda *values: jnp.concatenate(values), *chunks)
    n = np.arange(1, 14)
    np.testing.assert_array_equal(history["area"], n * (n + 1) / 2)
    np.testing.assert_array_equal(history["diagnostic"], n * (n + 3) / 2)
    assert float(final[0]["Area"][0, 0]) == 91 and float(final[1]) == 104
    assert set(history) == {"area", "diagnostic"}
    assert all(array.shape == (13,) for array in history.values())
    with h5netcdf.File(path, "r") as file:
        instant, mean = file.groups["instant"], file.groups["mean"]
        np.testing.assert_array_equal(
            instant.variables["time"][:], [0, 2, 4, 6, 8, 10, 12]
        )
        np.testing.assert_array_equal(
            instant.variables["Area"][:, 0, 0], [0, 3, 10, 21, 36, 55, 78]
        )
        np.testing.assert_array_equal(
            mean.variables["Area"][:, 0, 0], [2.5, 18.5, 50.5]
        )
        np.testing.assert_array_equal(
            mean.variables["time_bounds"][:], [[0, 4], [4, 8], [8, 12]]
        )


@pytest.mark.parametrize("scheduled", [False, True])
def test_requested_histories_bound_chunk_storage_without_stacking_unselected_fields(
    tmp_path: Path, scheduled: bool
) -> None:
    chunks = []
    settings = OutputSettings(
        streams=(
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=1000)),
        )
    )
    with OutputManager(tmp_path / "partial.nc", settings) as manager:
        observer, _ = output_callbacks(manager, 1) if scheduled else (None, None)
        final, _, _ = run_timed(
            _pair(),
            _advance,
            13,
            inputs=jnp.arange(1, 14, dtype=jnp.float32),
            observe=observer,
            select=lambda pair: pair[0],
            history_select=_select_history,
            history_sink=chunks.append,
        )
    assert [chunk["area"].shape for chunk in chunks] == [(8,), (5,)]
    assert all(set(chunk) == {"area", "diagnostic"} for chunk in chunks)
    assert all(array.ndim == 1 for chunk in chunks for array in chunk.values())
    assert final[0]["Area"].shape == (6, 7) and final[1].shape == ()
    assert not (tmp_path / "partial.nc").exists()


def test_default_reducer_with_dynamic_inputs_returns_only_constant_size_carry() -> None:
    """No requested history may add ys or timestep axes to the default reducer."""
    from veris.io.scan import _initial_carry, _reduce_chunk

    streams = (
        Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=1000)),
    )
    select = lambda pair: pair[0]
    shapes = []
    with jax.enable_x64():
        carry = _initial_carry(_pair(), streams, select)
        reduce = partial(
            _reduce_chunk,
            advance=_advance,
            select=select,
            streams=streams,
            intervals=(1,),
            sample_initial=True,
            checkpoint=False,
            physics_x64=True,
        )
        for length in (10, 100):
            inputs = jnp.ones((length,), dtype=jnp.float32)
            result = jax.eval_shape(partial(reduce, steps=length), carry, inputs=inputs)
            assert isinstance(result, tuple) and len(result) == 4
            shapes.append([leaf.shape for leaf in jax.tree.leaves(result)])
            graph = jax.make_jaxpr(partial(reduce, steps=length))(carry, inputs=inputs)
            scan = next(eq for eq in graph.jaxpr.eqns if eq.primitive.name == "scan")
            if "ft_out" in scan.params:
                output_carry, history = scan.params["ft_out"].unpack()
                assert not history and len(scan.outvars) == len(output_carry)
            else:
                assert len(scan.outvars) == scan.params["num_carry"]
        assert shapes[0] == shapes[1]


def _definition(output: OutputSettings | None, steps: int = 8) -> Any:
    return veris.Experiment.builtin(
        "growth",
        settings={"dtype": "float64", "deltatTherm": 1},
        forcing=veris.Forcing(series={"uWind": np.arange(3.0, 3.0 + steps)}),
        steps=steps,
        output=output,
    )


def _observe(state: Any, diagnostic: Any) -> dict[str, jax.Array]:
    return {"wind": state.uWind[2:-2, 2:-2], "freshwater": diagnostic.EmPmR[2:-2, 2:-2]}


def test_facade_split_means_survive_observations_enabled_disabled_enabled(
    tmp_path: Path,
) -> None:
    output = OutputSettings(
        streams=(
            Stream("instant", ("uWind",), timedelta(seconds=2)),
            Stream("mean", ("uWind",), timedelta(seconds=1), timedelta(seconds=4)),
        )
    )
    definition = _definition(output)
    with veris.Simulation.initialize(replace(definition, output=None)) as oracle:
        expected = oracle.run(8, observe=_observe)
    path = tmp_path / "split.nc"
    with veris.Simulation.initialize(definition, output_path=path) as simulation:
        first = simulation.run(3, observe=_observe)
        middle = simulation.run(2)
        last = simulation.run(3, observe=_observe)
    assert middle.observations is None and last.elapsed == timedelta(seconds=8)
    for name in ("wind", "freshwater"):
        np.testing.assert_array_equal(
            first.observations[name], expected.observations[name][:3]
        )
        np.testing.assert_array_equal(
            last.observations[name], expected.observations[name][5:]
        )
    np.testing.assert_array_equal(first.observations["wind"][:, 0, 0], [3, 4, 5])
    np.testing.assert_array_equal(last.observations["wind"][:, 0, 0], [8, 9, 10])
    with h5netcdf.File(path, "r") as file:
        np.testing.assert_array_equal(
            file.groups["mean"].variables["uWind"][:, 0, 0], [3.75, 7.5]
        )
        np.testing.assert_array_equal(
            file.groups["instant"].variables["time"][:], [0, 2, 4, 6, 8]
        )


def test_observed_zero_runs_keep_empty_shapes_after_forcing_exhaustion_and_no_duplicate_initial(
    tmp_path: Path,
) -> None:
    output = OutputSettings(
        streams=(Stream("instant", ("uWind",), timedelta(seconds=1)),)
    )
    path = tmp_path / "zero.nc"
    with veris.Simulation.initialize(
        _definition(output, steps=1), output_path=path
    ) as simulation:
        before = simulation.run(0, observe=_observe)
        state = simulation.state
        assert before.diagnostics is None and before.observations["wind"].shape == (
            0,
            2,
            2,
        )
        assert simulation.state is state
        simulation.run(1)
        completed = simulation.state
        after = simulation.run(0, observe=_observe)
        assert after.observations["freshwater"].shape == (0, 2, 2)
        assert simulation.state is completed and after.completed_steps == 1
    with h5netcdf.File(path, "r") as file:
        np.testing.assert_array_equal(
            file.groups["instant"].variables["time"][:], [0, 1]
        )


def test_observer_tracing_failure_closes_output_without_initial_file_effects(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failure.nc"
    output = OutputSettings(
        streams=(Stream("instant", ("uWind",), timedelta(seconds=1)),)
    )
    simulation = veris.Simulation.initialize(_definition(output), output_path=path)
    state = simulation.state

    def fail(state: Any, diagnostic: Any) -> Any:
        raise RuntimeError("observer trace failed")

    with pytest.raises(RuntimeError, match="observer trace failed"):
        simulation.run(2, observe=fail)
    assert simulation.closed and simulation.state is state
    assert simulation.completed_steps == 0 and simulation.elapsed == timedelta(0)
    assert not path.exists()
