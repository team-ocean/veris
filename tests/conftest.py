"""Small JAX PyTree fixtures and deterministic pre-fixture test selection."""

import hashlib
import importlib
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import make_dataclass
from functools import cache
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.compilation_cache import compilation_cache
from jax.typing import ArrayLike

from veris.configuration import Configuration
from veris.physical_constants import PhysicalConstants

jax.config.update("jax_enable_x64", True)


@contextmanager
def _temporary_compilation_cache() -> Iterator[None]:
    """Reuse costly compilations in this run and its children, then remove them.

    A private fresh directory allows reuse across independently bound rollout
    functions without relying on a warm cache from previous test sessions.
    Preserve caller settings and JAX's minimum compile-time/entry-size policy.
    """
    previous = jax.config.jax_compilation_cache_dir
    if (
        previous is not None
        or "JAX_COMPILATION_CACHE_DIR" in os.environ
        or not jax.config.jax_enable_compilation_cache
    ):
        yield
        return
    with (
        TemporaryDirectory(prefix="veris-pytest-jax-") as directory,
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setenv("JAX_COMPILATION_CACHE_DIR", directory)
        jax.config.update("jax_compilation_cache_dir", directory)
        try:
            yield
        finally:
            compilation_cache.reset_cache()
            jax.config.update("jax_compilation_cache_dir", previous)


@pytest.fixture(scope="session", autouse=True)
def compilation_cache_session() -> Iterator[None]:
    """Keep compilation reuse inside the test harness and preserve cold imports."""
    with _temporary_compilation_cache():
        yield


type StateFieldInput = ArrayLike | list[StateFieldInput] | tuple[StateFieldInput, ...]


class StateFactory(Protocol):
    """Build partial PyTrees whose field names are chosen independently per test.

    The dynamic dataclass result is the intentional Any boundary: each test
    supplies a different set of fields, so no single structural state protocol
    describes every result. Inputs include nested Python lists accepted by
    jnp.asarray, as well as NumPy/JAX arrays and numerical scalars.
    """

    def __call__(self, **fields: StateFieldInput) -> Any: ...


@cache
def state_type(fields: tuple[str, ...]) -> StateFactory:
    """Reuse PyTree node types to avoid compilation for identical structures."""
    # The generated class accepts exactly the runtime-selected keyword names.
    return cast(
        StateFactory,
        jax.tree_util.register_dataclass(
            make_dataclass("State", [(name, jax.Array) for name in fields], frozen=True)
        ),
    )


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
def conf() -> Configuration:
    """Return hashable source settings accepted by static JIT arguments."""
    return Configuration(use_sharding=False)


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
def phys() -> PhysicalConstants:
    """Return immutable physical constants independently of execution settings."""
    return PhysicalConstants()


@pytest.fixture
def halo() -> ModuleType:
    """Import periodic halo helpers; callers select execution through settings."""
    return importlib.import_module("veris.fill_overlap")
