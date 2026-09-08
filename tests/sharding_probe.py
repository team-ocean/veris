"""Fresh-process four-CPU probe of real periodic halo communication.

Each local block stores its own two-cell halos. The independent oracle indexes
one unpartitioned global interior modulo its dimensions; it never exchanges
neighbor buffers using the production algorithm.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from veris.settings import settings

settings["use_sharding"] = False
from veris.fill_overlap import fill_overlap_shard


def check_layout(px, py):
    """Compare every interior/edge/corner of each shard with global coordinates."""
    nx, ny = 3, 5
    gx, gy = px * nx, py * ny
    global_field = np.arange(gx * gy, dtype=float).reshape(gx, gy)
    packed = np.full((px * (nx + 4), py * (ny + 4)), -999.0)
    expected = np.empty_like(packed)
    for rank_x in range(px):
        for rank_y in range(py):
            block_x, block_y = rank_x * (nx + 4), rank_y * (ny + 4)
            packed[block_x + 2 : block_x + 2 + nx, block_y + 2 : block_y + 2 + ny] = (
                global_field[
                    rank_x * nx : (rank_x + 1) * nx, rank_y * ny : (rank_y + 1) * ny
                ]
            )
            for i, j in np.ndindex((nx + 4, ny + 4)):
                expected[block_x + i, block_y + j] = global_field[
                    (rank_x * nx + i - 2) % gx, (rank_y * ny + j - 2) % gy
                ]
    mesh = jax.make_mesh((px, py), ("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))
    data = jax.device_put(jnp.asarray(packed), sharding)
    fill = jax.shard_map(
        fill_overlap_shard, mesh=mesh, in_specs=P("x", "y"), out_specs=P("x", "y")
    )
    actual = np.asarray(fill(data))
    if not np.array_equal(actual, expected):
        location = np.unravel_index(np.argmax(np.abs(actual - expected)), actual.shape)
        raise AssertionError(
            f"ERROR layout {px}x{py} at {location}: {actual[location]} != {expected[location]}"
        )
    # A halo refresh is an idempotent projection, including the reverse pass.
    refreshed = fill(data)
    np.testing.assert_array_equal(fill(refreshed), expected)
    with jax.set_mesh(mesh):
        gradient = jax.grad(lambda value: jnp.sum(fill(value)))(data)
    counts = np.zeros_like(packed)
    for rank_x in range(px):
        for rank_y in range(py):
            for i, j in np.ndindex((nx + 4, ny + 4)):
                x, y = (rank_x * nx + i - 2) % gx, (rank_y * ny + j - 2) % gy
                counts[
                    (x // nx) * (nx + 4) + 2 + x % nx, (y // ny) * (ny + 4) + 2 + y % ny
                ] += 1
    np.testing.assert_array_equal(gradient, counts)


if __name__ == "__main__":
    assert len(jax.devices()) == 4, "ERROR expected four CPU devices"
    for layout in [(2, 2), (1, 4), (4, 1)]:
        check_layout(*layout)
    print("three layouts: halo values and reverse-mode gradients passed")
