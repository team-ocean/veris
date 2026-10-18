"""Physical base leaves remain dynamic without allocating devices in definitions."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris import Execution, Experiment, Grid, Simulation, initialize_experiment
from veris.execution.advance import advance
from veris.model.physical_constants import PhysicalConstants


def test_parameter_update_refreshes_traced_reciprocal() -> None:
    p = PhysicalConstants(h0=0.5)

    def reciprocal(x: jax.Array) -> jax.Array:
        return jnp.asarray(p.with_overrides(h0=x).recip_h0)

    np.testing.assert_allclose(jax.grad(reciprocal)(0.5), -4.0, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(reciprocal)(0.4), 2.5, rtol=1e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_base_leaf_round_trip_and_density_angle_dependencies(dtype: str) -> None:
    p = PhysicalConstants(
        dtype=dtype, rhoFresh=997.0, rhoIce=910.0, waterTurnAngle=30.0
    )
    names = p.base_parameter_names()
    assert "rhoFresh" in names and "h0" in names
    assert not {"dtype", "recip_h0", "sinWat", "rhoIce2rhoFresh"} & set(names)
    leaves, tree = jax.tree.flatten(p)
    assert all(np.asarray(leaf).dtype == np.dtype(dtype) for leaf in leaves)
    q = jax.tree.unflatten(tree, leaves)
    np.testing.assert_allclose(
        q.recip_rhoFresh, 1 / 997, rtol=float(4 * np.finfo(dtype).eps)
    )
    np.testing.assert_allclose(
        q.rhoIce2rhoFresh, 910 / 997, rtol=float(4 * np.finfo(dtype).eps)
    )
    np.testing.assert_allclose(q.sinWat, 0.5, rtol=float(4 * np.finfo(dtype).eps))
    np.testing.assert_allclose(
        q.cosWat, np.sqrt(3) / 2, rtol=float(4 * np.finfo(dtype).eps)
    )

    def derived(density: jax.Array) -> jax.Array:
        return jnp.asarray(p.with_overrides(rhoFresh=density).rhoIce2rhoFresh)

    np.testing.assert_allclose(
        jax.grad(derived)(jnp.asarray(997.0, dtype=dtype)),
        -910 / 997**2,
        rtol=float(8 * np.finfo(dtype).eps),
    )


def test_parameter_override_owns_concrete_input_and_preserves_original() -> None:
    p = PhysicalConstants(h0=0.5)
    authored = np.asarray(0.4)
    changed = p.with_overrides(h0=authored)
    authored[...] = 0.7
    assert p.h0 == 0.5
    assert float(changed.h0) == 0.4
    assert float(changed.recip_h0) == 2.5


@pytest.mark.parametrize(
    "parameters",
    [
        {"missing": 0.4},
        {"dtype": "float32"},
        {"recip_h0": 2.5},
        {"h0": -1.0},
        {"rhoFresh": 0.0},
        {"h0": np.nan},
        {"h0": np.ones(2)},
        {"h0": 1j},
        {"h0": True},
    ],
)
def test_invalid_or_derived_parameter_override_is_rejected(
    parameters: dict[str, Any],
) -> None:
    with pytest.raises((ValueError, TypeError)):
        PhysicalConstants().with_overrides(**parameters)


def test_same_schema_physical_values_trace_once() -> None:
    traces = []
    p = PhysicalConstants()

    @jax.jit
    def evaluate(value: jax.Array) -> jax.Array:
        traces.append(1)
        return jnp.asarray(p.with_overrides(h0=value).recip_h0)

    np.testing.assert_allclose(evaluate(jnp.asarray(0.4)), 2.5)
    np.testing.assert_allclose(evaluate(jnp.asarray(0.5)), 2.0)
    assert len(traces) == 1


def test_definition_and_json_remain_device_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ERROR device discovery while constructing definition")

    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(jax, "local_devices", forbidden)
    monkeypatch.setattr(jnp, "asarray", forbidden)
    definition = Experiment.builtin("growth", constants={"h0": 0.43, "rhoFresh": 997.0})
    from veris.model.experiment_file import experiment_from_dict

    restored = experiment_from_dict(definition.to_dict())
    assert restored.physical_constants.h0 == 0.43
    assert restored.physical_constants.rhoFresh == 997.0


def test_invalid_traced_parameter_returns_status_and_host_rejects_candidate() -> None:
    definition = Experiment.builtin("growth", grid=Grid(2, 2), steps=1)
    state, conf, p = initialize_experiment(definition)
    invalid = jax.jit(lambda x: p.with_overrides(h0=x))(jnp.asarray(-0.5))
    result, diagnostics = jax.jit(
        lambda s, c: advance(s, conf, c, physics="thermodynamics")
    )(state, invalid)
    np.testing.assert_array_equal(diagnostics.physics_status, 1.0)
    for actual, expected in zip(
        jax.tree.leaves(result), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    simulation = Simulation(definition, state, conf, invalid, None)
    with pytest.raises(ValueError, match="physical"):
        simulation.run(1)
    assert simulation.state is state
    assert simulation.completed_steps == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_geometry_precision_is_validated_before_device_selection(
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
) -> None:
    """Rounding a valid latitude to a singular pole must fail on the host."""
    import runpy
    from dataclasses import replace
    from pathlib import Path

    make_geometry = runpy.run_path(str(Path("tests/grid/test_spherical_operators.py")))[
        "spherical_geometry"
    ]
    geometry = make_geometry(np.arange(82.0, 90.0), halos=False)
    geometry = replace(
        geometry,
        latitude=np.r_[np.arange(82.0, 89.0), 89.999998],
        latitude_edges=np.r_[81.5, np.arange(82.5, 89.0), 89.999999],
    )
    definition = Experiment(
        grid=Grid(8, 8),
        geometry=geometry,
        settings={"dtype": dtype},
        execution=Execution(backend="cpu"),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ERROR device selection reached")

    monkeypatch.setattr(jax, "devices", forbidden)
    if dtype == "float32":
        with pytest.raises(ValueError, match="latitude|singular"):
            initialize_experiment(definition)
    else:
        with pytest.raises(AssertionError, match="device selection reached"):
            initialize_experiment(definition)


def test_parameter_pytree_allows_jax_placeholder_reconstruction() -> None:
    """Sharding/transform schema trees must not evaluate reciprocal laws on None."""
    p = PhysicalConstants()
    symbolic = jax.tree.map(lambda _: None, p)
    assert isinstance(symbolic, PhysicalConstants)
    assert symbolic.dtype == "float64"
    assert jax.tree.leaves(symbolic) == []


def test_traced_override_refuses_silent_precision_truncation() -> None:
    """A static float64 schema cannot silently contain traced float32 leaves."""
    p = PhysicalConstants(dtype="float64")
    with jax.enable_x64(False), pytest.raises(ValueError, match="dtype|x64|precision"):
        jax.jit(lambda x: p.with_overrides(h0=x).recip_h0)(jnp.asarray(0.4))


def test_parameter_tree_input_refuses_silent_precision_truncation() -> None:
    """Direct dynamic parameter arguments retain the same precision contract."""
    p = PhysicalConstants(dtype="float64")
    with jax.enable_x64(False), pytest.raises(ValueError, match="dtype|x64|precision"):
        jax.jit(lambda parameters: parameters.recip_h0)(p)


def test_host_parameter_tree_round_trip_does_not_create_device_values() -> None:
    """Host tree utilities keep definition parameters on the host before x64 is enabled."""
    p = PhysicalConstants(h0=0.43)
    values, tree = jax.tree.flatten(p)
    with jax.enable_x64(False):
        rebuilt = jax.tree.unflatten(tree, values)
    assert all(
        not isinstance(getattr(rebuilt, name), jax.Array)
        for name in ("recip_h0", "sinWat", "cosWat")
    )
    assert rebuilt.recip_h0 == 1 / 0.43
