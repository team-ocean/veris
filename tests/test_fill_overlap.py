"""Periodic halos must reproduce wrapped interior values, including corners."""

from functools import partial
from types import ModuleType

import jax.numpy as jnp
import numpy as np
import pytest

from veris.configuration import Settings


@pytest.mark.parametrize("shape", [(2, 3), (4, 7), (6, 4)])
def test_periodic_halo_matches_numpy_wrap(
    halo: ModuleType, shape: tuple[int, int]
) -> None:
    interior = np.arange(np.prod(shape), dtype=float).reshape(shape)
    initial = np.pad(interior, 2, constant_values=-999)
    expected = np.pad(interior, 2, mode="wrap")
    for fill in (
        halo.fill_circular_overlap,
        partial(halo.fill_overlap, sett=Settings(use_sharding=False)),
    ):
        actual = fill(jnp.asarray(initial))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(fill(actual), expected)
    u, v = halo.fill_overlap_uv(
        jnp.asarray(initial), jnp.asarray(-initial), Settings(use_sharding=False)
    )
    np.testing.assert_array_equal(u, expected)
    np.testing.assert_array_equal(v, -expected)


def test_shard_map_halo_matches_serial_on_one_device(halo: ModuleType) -> None:
    """Execute actual collective halo code on the available single-device mesh."""
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((1, 1), ("x", "y"))
    interior = np.arange(28.0).reshape(4, 7)
    initial = jnp.asarray(np.pad(interior, 2, constant_values=-99))
    initial = jax.device_put(initial, NamedSharding(mesh, P("x", "y")))
    fill = halo.make_sharded_fill_overlap(mesh)
    np.testing.assert_array_equal(fill(initial), np.pad(interior, 2, mode="wrap"))


def test_four_cpu_halo_exchange_and_adjoint_in_fresh_process() -> None:
    """Exercise real cross-device communication in all rectangular mesh layouts."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", PYTHONPATH=str(root)
    )
    result = subprocess.run(
        [sys.executable, str(root / "tests/sharding_probe.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR halo communication probe: {result.stderr[-2000:]}"
    )
    assert "halo values and reverse-mode gradients passed" in result.stdout


def test_sharded_dispatch_uses_explicit_active_mesh(halo: ModuleType) -> None:
    """Settings select exchange at call time and the caller supplies the mesh."""
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((1, 1), ("x", "y"))
    interior = np.arange(12.0).reshape(3, 4)
    data = jnp.asarray(np.pad(interior, 2, constant_values=-99))
    data = jax.device_put(data, NamedSharding(mesh, P("x", "y")))
    expected = np.pad(interior, 2, mode="wrap")
    with jax.set_mesh(mesh):
        actual = halo.fill_overlap(data, Settings(use_sharding=True))
        u, v = halo.fill_overlap_uv(data, -data, Settings(use_sharding=True))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(u, expected)
    np.testing.assert_array_equal(v, -expected)
    np.testing.assert_array_equal(
        halo.fill_overlap(
            np.pad(interior, 2, constant_values=-99), Settings(use_sharding=False)
        ),
        expected,
    )


def test_sharded_dispatch_requires_named_mesh(halo: ModuleType) -> None:
    """Missing or unrelated meshes fail at the host boundary with useful errors."""
    import jax

    data = jnp.zeros((8, 8))
    with pytest.raises(ValueError, match="mesh"):
        halo.fill_overlap(data, Settings(use_sharding=True))
    mesh = jax.make_mesh((1,), ("devices",))
    with pytest.raises(ValueError, match="x.*y"):
        halo.make_sharded_fill_overlap(mesh)


def test_halo_import_does_not_require_application_mesh_module() -> None:
    """Import must succeed in a fresh process with default sharding enabled."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import veris.fill_overlap"],
        env=dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR standalone halo import: {result.stderr[-2000:]}"
    )
