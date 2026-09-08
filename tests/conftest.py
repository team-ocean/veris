"""Small JAX PyTree fixtures and deterministic pre-fixture test selection."""

import hashlib
import importlib
import math
import os
from collections import namedtuple
from functools import cache

import jax
import jax.numpy as jnp
import pytest

from veris.settings import settings

jax.config.update("jax_enable_x64", True)
Settings = namedtuple("Settings", settings)


@cache
def state_type(fields):
    """Reuse PyTree node types to avoid compilation for identical structures."""
    return namedtuple("State", fields)


def pytest_addoption(parser):
    """Expose the project-wide development subsample."""
    parser.addoption("--fast", action="store_true", help="Run a stable 10% sample")


def pytest_collection_modifyitems(config, items):
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
def sett():
    """Return hashable source settings accepted by static JIT arguments."""
    return Settings(**settings)


@pytest.fixture
def state():
    """Build a PyTree containing only the fields required by each kernel."""

    def build(**fields):
        return state_type(tuple(fields))(
            **{
                key: jnp.asarray(value, dtype=jnp.float64)
                for key, value in fields.items()
            }
        )

    return build


@pytest.fixture
def halo(monkeypatch):
    """Select the standalone serial mode without an external mesh module."""
    from veris.settings import settings

    monkeypatch.setitem(settings, "use_sharding", False)
    return importlib.import_module("veris.fill_overlap")
