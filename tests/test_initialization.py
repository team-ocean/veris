"""Initialization builds every numerical field before any compiled calculation."""

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.state import State


def test_initialize_allocates_complete_minimal_frozen_state() -> None:
    """Registry metadata controls defaults, shapes and the exact PyTree leaves."""
    from veris.initialization import initialize
    from veris.variables import VARIABLES

    state, settings, constants = initialize(nx=4, ny=7)
    assert all(is_dataclass(value) for value in (state, settings, constants))
    assert tuple(field.name for field in fields(state)) == tuple(VARIABLES)
    assert len(jax.tree.leaves(state)) == len(VARIABLES)
    for name, metadata in VARIABLES.items():
        array = getattr(state, name)
        assert isinstance(array, jax.Array)
        assert array.shape == (8, 11)
        assert array.dtype == np.dtype(metadata.dtype)
        np.testing.assert_array_equal(array, metadata.default)
    with pytest.raises(FrozenInstanceError):
        setattr(state, "theta", jnp.zeros((8, 11)))  # noqa: B010


def test_dataclass_state_supports_jit_jvp_vjp_and_replacement() -> None:
    """Registration preserves numerical differentiation and immutable updates."""
    from veris.initialization import initialize

    state, _, _ = initialize(nx=4, ny=5)
    changed = replace(state, hIceMean=jnp.full_like(state.hIceMean, 2.0))
    tangent = jax.tree.map(jnp.zeros_like, changed)
    tangent = replace(tangent, hIceMean=jnp.ones_like(changed.hIceMean))

    def energy(value: State) -> jax.Array:
        return jnp.sum(value.hIceMean**2)

    primal, derivative = jax.jvp(jax.jit(energy), (changed,), (tangent,))
    np.testing.assert_allclose(primal, 4 * 8 * 9)
    np.testing.assert_allclose(derivative, 4 * 8 * 9)
    gradient = jax.grad(energy)(changed)
    np.testing.assert_array_equal(gradient.hIceMean, 4)
    np.testing.assert_array_equal(gradient.theta, 0)
    np.testing.assert_array_equal(state.hIceMean, 0)


@pytest.mark.parametrize(
    "nx,ny", [(0, 5), (5, -1), (1, 5), (5, 1), (True, 5), (4.5, 5)]
)
def test_invalid_grid_is_rejected(nx: object, ny: object) -> None:
    """Host validation catches invalid dimensions before allocating arrays."""
    from veris.initialization import initialize

    with pytest.raises((TypeError, ValueError), match="nx|ny|grid"):
        initialize(nx=nx, ny=ny)  # ty: ignore[invalid-argument-type]


def test_initializer_accepts_separate_configuration_and_array_overrides() -> None:
    """Overrides remain distinct and cannot introduce extra AD leaves."""
    from veris.initialization import initialize

    state, settings, constants = initialize(
        nx=4,
        ny=5,
        settings_overrides={"deltatDyn": 600.0},
        physical_overrides={"rhoIce": 920.0},
        state_overrides={"hIceMean": jnp.ones((8, 9))},
    )
    assert settings.recip_deltatDyn == 1 / 600
    assert constants.rhoIce == 920
    np.testing.assert_array_equal(state.hIceMean, 1)
    assert not hasattr(state, "settings") and not hasattr(settings, "rhoIce")


def test_initializer_rejects_unknown_fields_and_mismatched_shapes() -> None:
    """Misspelled metadata keys and incomplete grid arrays cannot enter State."""
    from veris.initialization import initialize

    with pytest.raises((TypeError, ValueError), match="unknown|not_a_field"):
        initialize(state_overrides={"not_a_field": 0})
    with pytest.raises(ValueError, match="hIceMean.*shape|shape.*hIceMean"):
        initialize(nx=4, ny=5, state_overrides={"hIceMean": jnp.ones((4, 5))})


def test_initializer_requires_declared_allocation_precision() -> None:
    """A disabled x64 configuration must not silently truncate registry dtypes."""
    from veris.initialization import initialize

    with (
        jax.enable_x64(False),
        pytest.raises(ValueError, match="jax_enable_x64"),
    ):
        initialize(nx=4, ny=5)


def test_initializer_rejects_nonnumeric_state_arrays() -> None:
    """Conversion errors identify the field before any compiled calculation."""
    from veris.initialization import initialize

    with pytest.raises(TypeError, match="hIceMean.*numeric"):
        initialize(nx=4, ny=5, state_overrides={"hIceMean": [["invalid"]]})


@pytest.mark.parametrize("mesh", ["x,y", object()])
def test_initializer_rejects_nonmesh_resources(mesh: object) -> None:
    """Host initialization must reject arbitrary resources before placement."""
    from veris.initialization import initialize

    with pytest.raises(TypeError, match="mesh.*jax.sharding.Mesh"):
        initialize(4, 5, mesh=mesh)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("axes", [("x", "z"), ("row", "column")])
def test_initializer_rejects_incompatible_mesh_axes(axes: tuple[str, str]) -> None:
    """Partition layout must expose the axes required by production halos."""
    from veris.initialization import initialize

    mesh = jax.make_mesh((1, 1), axes)
    with pytest.raises(ValueError, match="mesh.*axes x and y"):
        initialize(4, 5, mesh=mesh)


def test_initializer_rejects_serial_settings_with_explicit_mesh() -> None:
    """A mesh must not silently change an explicitly selected halo backend."""
    from veris.initialization import initialize

    mesh = jax.make_mesh((1, 1), ("x", "y"))
    with pytest.raises(ValueError, match="use_sharding=True"):
        initialize(4, 5, mesh=mesh, settings_overrides={"use_sharding": False})


def test_mesh_initializer_rejects_overrides_missing_halos() -> None:
    """Array overrides must obey the same halo-inclusive mesh storage contract."""
    from veris.initialization import initialize

    mesh = jax.make_mesh((1, 1), ("x", "y"))
    with pytest.raises(ValueError, match="hIceMean.*shape.*halo-inclusive"):
        initialize(4, 5, mesh=mesh, state_overrides={"hIceMean": np.ones((4, 5))})
