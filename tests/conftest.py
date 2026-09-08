"""Small JAX PyTree fixtures and deterministic pre-fixture test selection."""

import hashlib
import importlib
import math
import os
from collections import namedtuple
from collections.abc import MutableMapping
from functools import cache
from types import ModuleType
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp
import pytest
from jax.typing import ArrayLike

from veris.settings import settings
from veris.state import Settings

jax.config.update("jax_enable_x64", True)

type StateFieldInput = ArrayLike | list[StateFieldInput] | tuple[StateFieldInput, ...]


class StateFactory(Protocol):
    """Build partial PyTrees whose field names are chosen independently per test.

    The dynamic namedtuple result is the intentional Any boundary: each test
    supplies a different set of fields, so no single structural state protocol
    describes every result. Inputs include nested Python lists accepted by
    jnp.asarray, as well as NumPy/JAX arrays and numerical scalars.
    """

    def __call__(self, **fields: StateFieldInput) -> Any: ...


@cache
def state_type(fields: tuple[str, ...]) -> StateFactory:
    """Reuse PyTree node types to avoid compilation for identical structures."""
    # The generated class accepts exactly the runtime-selected keyword names.
    return cast(StateFactory, namedtuple("State", fields))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose the project-wide development subsample."""
    parser.addoption("--fast", action="store_true", help="Run a stable 10% sample")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Select before fixtures; allow agents to vary the deterministic seed."""
    if not config.getoption("--fast") or not items:
        return
    seed = os.environ.get("VERIS_TEST_SEED", "root")
    ranked = sorted(
        items,
        key=lambda item: hashlib.sha256(f"{seed}:{item.nodeid}".encode()).digest(),
    )
    selected = set(ranked[: math.ceil(len(items) / 10)])
    config.hook.pytest_deselected(items=[i for i in items if i not in selected])
    items[:] = [i for i in items if i in selected]


@pytest.fixture
def sett() -> Settings:
    """Return hashable source settings accepted by static JIT arguments."""
    return Settings(**settings)


@pytest.fixture
def state() -> StateFactory:
    """Build a PyTree containing only the fields required by each kernel."""

    def build(**fields: StateFieldInput) -> Any:
        """Convert dynamic fields to float64 arrays before PyTree construction."""
        return state_type(tuple(fields))(
            **{
                key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in fields.items()
            }
        )

    return build


@pytest.fixture
def halo(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Select the standalone serial mode without an external mesh module."""
    from veris.settings import settings

    # pytest's generic Mapping signature needs the heterogeneous value type.
    registry = cast(MutableMapping[str, bool | float], settings)
    monkeypatch.setitem(registry, "use_sharding", False)
    return importlib.import_module("veris.fill_overlap")
