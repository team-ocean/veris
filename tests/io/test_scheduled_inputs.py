"""Host timed scans accept run-local dynamic inputs without changing output.

Prefix sums of distinct forcing
values are independent oracles for chunk/event slicing; no numerical probes or
pytest runs were performed while preparing this file.
"""

from datetime import timedelta
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.execution.rollout_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream

_DELTAS = np.array([1, 10, 100, 1000, 10000, 100000, 1000000], dtype=np.float32)
_BIAS = np.array([2, 3, 5, 7, 11, 13, 17], dtype=np.float32)
_PREFIX = [0, 1, 11, 111, 1111, 11111, 111111, 1111111]


def _initial() -> dict[str, jax.Array]:
    """A small rectangular halo-bearing carry with an unselected field."""
    return {
        "Area": jnp.zeros((6, 7), dtype=jnp.float32),
        "hIceMean": jnp.full((6, 7), 5, dtype=jnp.float32),
    }


def _inputs(start: int = 0, stop: int = 7) -> dict[str, Any]:
    """Use a nested PyTree so every array leaf must be sliced consistently."""
    return {
        "delta": jnp.asarray(_DELTAS[start:stop]),
        "extra": {"bias": jnp.asarray(_BIAS[start:stop])},
    }


def _advance(
    state: dict[str, jax.Array], forcing: dict[str, Any]
) -> dict[str, jax.Array]:
    """Add independently checkable prescribed values to two physical fields."""
    return {
        "Area": state["Area"] + forcing["delta"],
        "hIceMean": state["hIceMean"] + 2 * forcing["delta"] + forcing["extra"]["bias"],
    }


def _settings() -> OutputSettings:
    """Mixed event lengths expose slices reused at different absolute positions."""
    return OutputSettings(
        streams=(
            Stream("instant", ("Area",), timedelta(seconds=2)),
            Stream("mean", ("Area",), timedelta(seconds=1), timedelta(seconds=3)),
        )
    )


def _assert_final(state: dict[str, jax.Array]) -> None:
    np.testing.assert_array_equal(state["Area"], 1111111)
    np.testing.assert_array_equal(state["hIceMean"], 2222285)
    assert state["Area"].dtype == jnp.float32


def _assert_records(path: Path) -> None:
    """Check independent endpoint and half-open mean oracles."""
    with h5netcdf.File(path, "r") as file:
        instant, mean = file.groups["instant"], file.groups["mean"]
        assert "hIceMean" not in instant.variables and "hIceMean" not in mean.variables
        np.testing.assert_array_equal(instant.variables["time"][:], [0, 2, 4, 6])
        np.testing.assert_array_equal(
            instant.variables["Area"][:, 0, 0], [0, 11, 1111, 111111]
        )
        np.testing.assert_array_equal(
            mean.variables["time_bounds"][:], [[0, 3], [3, 6]]
        )
        np.testing.assert_array_equal(mean.variables["sample_count"][:], [3, 3])
        np.testing.assert_array_equal(mean.variables["Area"][:, 0, 0], [4, 4111])
        assert instant.variables["Area"].dtype == np.dtype("float32")
        assert mean.variables["Area"].dtype == np.dtype("float64")


@pytest.mark.parametrize("checkpoint", [False, True])
def test_unobserved_timed_inputs_keep_the_final_carry_only(checkpoint: bool) -> None:
    final, compilation, integration = run_timed(
        _initial(), _advance, 7, inputs=_inputs(), checkpoint=checkpoint, chunk_size=2
    )
    assert isinstance(final, dict)
    _assert_final(final)
    assert compilation >= 0 and integration >= 0


def test_none_inputs_preserve_unary_transition_calls() -> None:
    """Supplying None explicitly retains the established unary advance contract."""
    final, _, _ = run_timed(
        _initial(),
        lambda state: {name: value + 1 for name, value in state.items()},
        2,
        inputs=None,
    )
    np.testing.assert_array_equal(final["Area"], 2)
    np.testing.assert_array_equal(final["hIceMean"], 7)


@pytest.mark.parametrize("chunk_size", [2, 3])
def test_generic_observer_inputs_slice_each_chunk_and_tail(chunk_size: int) -> None:
    """The generic host callback sees selected initial and post-step values."""
    observed = []

    def observe(sample: dict[str, jax.Array], iteration: int) -> None:
        assert set(sample) == {"Area"}
        observed.append((iteration, float(sample["Area"][0, 0])))

    final, _, _ = run_timed(
        _initial(),
        _advance,
        7,
        inputs=_inputs(),
        chunk_size=chunk_size,
        select=lambda state: {"Area": state["Area"]},
        observe=observe,
    )
    _assert_final(final)
    assert observed == list(enumerate(_PREFIX))


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_scheduled_inputs_follow_event_slices_and_preserve_half_open_means(
    tmp_path: Path, chunk_size: int | None
) -> None:
    path = tmp_path / "scheduled.nc"
    with OutputManager(path, _settings()) as manager:
        observe, select = output_callbacks(manager, 1)
        final, _, _ = run_timed(
            _initial(),
            _advance,
            7,
            inputs=_inputs(),
            observe=observe,
            select=select,
            chunk_size=chunk_size,
        )
    _assert_final(final)
    _assert_records(path)


def test_scheduled_pair_carry_keeps_final_diagnostics_separate_with_state_selector(
    tmp_path: Path,
) -> None:
    """A State/Diagnostics pair requires neither has_aux nor a full State history."""
    initial = (_initial(), jnp.array(-1, dtype=jnp.float32))

    def advance(
        carry: tuple[dict[str, jax.Array], jax.Array], forcing: dict[str, Any]
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        state, _ = carry
        updated = _advance(state, forcing)
        diagnostic = updated["Area"][0, 0] + forcing["extra"]["bias"]
        return updated, diagnostic

    path = tmp_path / "pair.nc"
    with OutputManager(path, _settings()) as manager:
        observe, _ = output_callbacks(manager, 1)
        final, _, _ = run_timed(
            initial,
            advance,
            7,
            inputs=_inputs(),
            observe=observe,
            select=lambda carry: carry[0],
        )
    assert isinstance(final, tuple) and len(final) == 2
    state, diagnostic = final
    _assert_final(state)
    assert diagnostic.shape == () and float(diagnostic) == 1111128
    _assert_records(path)


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize(
    "bad_inputs",
    [
        {},
        np.array(1, dtype=np.float32),
        np.ones((6,), dtype=np.float32),
        np.ones((8,), dtype=np.float32),
        {"delta": np.ones(7), "extra": {"bias": np.ones(6)}},
        {"delta": np.ones(7), "extra": {"bias": 1.0}},
    ],
)
def test_invalid_input_leaves_fail_before_observations_or_manager_reservation(
    tmp_path: Path, scheduled: bool, bad_inputs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation precedes compilation, collection, samples and reduced ownership."""
    effects = []
    path = tmp_path / "invalid.nc"
    with OutputManager(path, _settings()) as manager:
        original_begin = manager.begin_reduced

        def begin(*args: Any, **kwargs: Any) -> None:
            effects.append("reserved")
            original_begin(*args, **kwargs)

        monkeypatch.setattr(manager, "begin_reduced", begin)
        observe, select = (
            output_callbacks(manager, 1)
            if scheduled
            else (
                lambda sample, iteration: effects.append("observed"),
                lambda state: state,
            )
        )
        with pytest.raises(
            ValueError, match="inputs.*leading|inputs.*length|array leaves"
        ):
            run_timed(
                _initial(),
                _advance,
                7,
                inputs=bad_inputs,
                observe=observe,
                select=select,
            )
        assert effects == [] and manager.buffer_nbytes == 0 and not path.exists()
        if scheduled:
            # A rejected request must leave this exact manager available to retry.
            final, _, _ = run_timed(
                _initial(),
                _advance,
                7,
                inputs=_inputs(),
                observe=observe,
                select=select,
            )
            _assert_final(final)


def test_continuation_inputs_are_local_to_each_run_despite_absolute_output_steps(
    tmp_path: Path,
) -> None:
    """Absolute event indices must not index a short run-local forcing sequence."""
    path = tmp_path / "continued.nc"
    with OutputManager(path, _settings()) as manager:
        observe, select = output_callbacks(manager, 1, continue_runs=True)
        state = _initial()
        for start, stop in ((0, 2), (2, 2), (2, 5), (5, 7)):
            state, _, _ = run_timed(
                state,
                _advance,
                stop - start,
                inputs=_inputs(start, stop),
                observe=observe,
                select=select,
            )
    _assert_final(state)
    _assert_records(path)


def test_zero_step_inputs_validate_without_executing_the_binary_transition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "zero.nc"
    with OutputManager(path, _settings()) as manager:
        observe, select = output_callbacks(manager, 1)
        final, _, _ = run_timed(
            _initial(),
            _advance,
            0,
            inputs=_inputs(0, 0),
            observe=observe,
            select=select,
        )
    np.testing.assert_array_equal(final["Area"], 0)
    np.testing.assert_array_equal(final["hIceMean"], 5)
    with h5netcdf.File(path, "r") as file:
        assert set(file.groups) == {"instant"}
        np.testing.assert_array_equal(file.groups["instant"].variables["time"][:], [0])
