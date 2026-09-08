"""Typed immutable containers must preserve registry order and JAX PyTree behavior."""

import jax
import jax.numpy as jnp
import numpy as np

from veris.settings import settings
from veris.variables import variables


def test_settings_schema_matches_registry() -> None:
    """Catch drift between explicit static fields and runtime configuration."""
    from veris.state import Settings

    assert Settings._fields == tuple(settings)
    actual = Settings._make(settings.values())
    assert actual._asdict() == settings
    assert hash(actual) == hash(tuple(settings.values()))


def test_state_preserves_namedtuple_pytree_and_replacement() -> None:
    """Changing container typing must not change field order or AD traversal."""
    from veris.state import State

    assert State._fields == (*variables, "forc_salt_surface")
    values = [jnp.full((2, 3), i, dtype=float) for i in range(len(State._fields))]
    state = State._make(values)
    leaves, definition = jax.tree.flatten(state)
    assert len(leaves) == len(values)
    rebuilt = jax.tree.unflatten(definition, leaves)
    assert isinstance(rebuilt, State)
    np.testing.assert_array_equal(rebuilt.hIceMean, values[0])
    changed = rebuilt._replace(hIceMean=jnp.ones((2, 3)))
    np.testing.assert_array_equal(rebuilt.hIceMean, values[0])
    np.testing.assert_array_equal(changed.hIceMean, 1)
