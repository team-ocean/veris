"""One initialization precision governs scalar coefficients and model calculations."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from veris.configuration import SETTINGS
from veris.initialization import initialize
from veris.physical_constants import PHYSICALCONSTANTS
from veris.setups import island


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_policy_types_every_array_and_static_coefficient(dtype: str) -> None:
    """Defaults, overrides, derived constants and tables share the selected dtype."""
    state, settings, constants = initialize(4, 5, dtype=dtype)
    assert settings.dtype == constants.dtype == dtype
    for array in jax.tree.leaves(state):
        assert array.dtype == np.dtype(dtype)
    for instance, registry in ((settings, SETTINGS), (constants, PHYSICALCONSTANTS)):
        for name, metadata in registry.items():
            value = getattr(instance, name)
            if metadata.type is float or metadata.type is tuple:
                assert np.asarray(value).dtype == np.dtype(dtype), name
    changed = replace(constants, rhoIce=920.0)
    assert changed.dtype == dtype
    assert np.asarray(changed.rhoIce2rhoSnow).dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_coupled_step_and_gradients_preserve_precision(dtype: str) -> None:
    """Real dynamics, growth, diagnostics and AD must not promote field precision."""
    state, settings, constants = island.initialize(4, 5, dtype=dtype)
    result, diagnostics = island.step_with_diagnostics(state, settings, constants)
    for array in jax.tree.leaves((state, result, diagnostics)):
        assert array.dtype == np.dtype(dtype)
        assert bool(jnp.all(jnp.isfinite(array)))

    def ice(cooling: jax.Array) -> jax.Array:
        return jnp.sum(
            island.compiled_step(state, settings, constants, cooling).hIceMean
        )

    gradient = jax.grad(ice)(jnp.asarray(100.0, dtype=dtype))
    assert gradient.dtype == np.dtype(dtype)
    assert bool(jnp.isfinite(gradient))


def test_float32_works_without_global_x64() -> None:
    """Selecting single precision does not require changing global JAX state."""
    with jax.enable_x64(False):
        state, settings, constants = initialize(4, 5, dtype="float32")
        assert state.theta.dtype == np.dtype("float32")
        assert settings.dtype == constants.dtype == "float32"


@pytest.mark.parametrize("dtype", ["float16", "int32", "complex64", "invalid"])
def test_invalid_precision_is_rejected(dtype: str) -> None:
    """Unsupported arithmetic must fail at the initialization boundary."""
    with pytest.raises(ValueError, match="dtype"):
        initialize(4, 5, dtype=dtype)


def test_precision_overflow_and_underflow_are_rejected() -> None:
    """Host-finite overrides must remain valid when rounded to model precision."""
    with pytest.raises(ValueError, match="rhoIce"):
        initialize(4, 5, dtype="float32", physical_overrides={"rhoIce": 1e100})
    with pytest.raises(ValueError, match="rhoIce"):
        initialize(4, 5, dtype="float32", physical_overrides={"rhoIce": 1e-100})


def test_derived_constant_underflow_is_rejected() -> None:
    """Representable base constants must not silently create a zero density ratio."""
    from veris.physical_constants import PhysicalConstants

    with pytest.raises(ValueError, match="rhoIce2rhoSnow"):
        PhysicalConstants(dtype="float32", rhoIce=1e-30, rhoSnow=1e30)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_mesh_allocation_preserves_precision_and_placement(dtype: str) -> None:
    """Explicit placement must retain the selected dtype on every state leaf."""
    mesh = jax.make_mesh((1, 1), ("x", "y"))
    state, settings, constants = initialize(2, 3, dtype=dtype, mesh=mesh)
    for array in jax.tree.leaves(state):
        assert array.dtype == np.dtype(dtype)
        assert array.sharding == jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x", "y")
        )
    assert settings.dtype == constants.dtype == dtype
