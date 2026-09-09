"""Typed immutable containers must preserve registry order and JAX PyTree behavior."""

from dataclasses import fields, is_dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from veris.variables import VARIABLES


def test_settings_schema_matches_registry() -> None:
    """Catch drift between explicit static fields and runtime configuration."""
    from veris.configuration import SETTINGS
    from veris.configuration import Settings as ModelSettings
    from veris.state import Settings

    assert Settings is ModelSettings
    actual = Settings()
    assert is_dataclass(actual)
    assert {field.name for field in fields(Settings)} == SETTINGS.keys()
    assert all(
        getattr(actual, name) == metadata.default for name, metadata in SETTINGS.items()
    )


def test_state_preserves_dataclass_pytree_and_replacement() -> None:
    """Changing container typing must not change field order or AD traversal."""
    from veris.state import State

    assert tuple(field.name for field in fields(State)) == tuple(VARIABLES)
    values = [jnp.full((2, 3), i, dtype=float) for i in range(len(VARIABLES))]
    state = State(**dict(zip(VARIABLES, values, strict=True)))
    leaves, definition = jax.tree.flatten(state)
    assert len(leaves) == len(values)
    rebuilt = jax.tree.unflatten(definition, leaves)
    assert isinstance(rebuilt, State)
    np.testing.assert_array_equal(rebuilt.hIceMean, values[0])
    changed = replace(rebuilt, hIceMean=jnp.ones((2, 3)))
    np.testing.assert_array_equal(rebuilt.hIceMean, values[0])
    np.testing.assert_array_equal(changed.hIceMean, 1)
