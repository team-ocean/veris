"""Real JAX distributed reductions with unequal local interiors and AD oracles.

Run as a worker with --rank/--count/--coordinator, or use launch_processes from
pytest. JAX_PLATFORMS selects CPU or CUDA before initialization. Packed local
blocks include poisoned halos; only their interiors belong in residual norms.
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, TextIO, cast


def launch_processes(
    platform: Literal["cpu", "cuda"] = "cpu",
    count: int = 2,
    devices_per_process: int = 1,
    timeout: float = 90,
) -> list[str]:
    """Launch local ranks, return concise output, and reap all ranks on failure."""
    root = Path(__file__).resolve().parents[1]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = None if visible is None else [item.strip() for item in visible.split(",")]
    if (
        platform == "cuda"
        and gpu_ids is not None
        and (len(gpu_ids) < count or any(item in ("", "-1") for item in gpu_ids))
    ):
        raise ValueError("ERROR not enough allocated CUDA devices for all ranks")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    processes: list[subprocess.Popen[bytes]] = []
    logs: list[TextIO] = []
    with (
        tempfile.TemporaryDirectory(prefix="veris-reductions-") as directory,
        ExitStack() as files,
    ):
        try:
            for rank in range(count):
                env = dict(os.environ, JAX_PLATFORMS=platform, PYTHONPATH=str(root))
                # Local communication must not go through an HTTP proxy.
                for key in tuple(env):
                    if key.lower().endswith("_proxy"):
                        env.pop(key)
                if platform == "cpu":
                    env["JAX_NUM_CPU_DEVICES"] = str(devices_per_process)
                    env["JAX_CPU_COLLECTIVES_IMPLEMENTATION"] = "gloo"
                else:
                    env["CUDA_VISIBLE_DEVICES"] = (
                        str(rank) if gpu_ids is None else gpu_ids[rank]
                    )
                log = files.enter_context(
                    open(Path(directory) / f"rank-{rank}.log", "w+")
                )
                logs.append(log)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--rank",
                            str(rank),
                            "--count",
                            str(count),
                            "--coordinator",
                            coordinator,
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            deadline = time.monotonic() + timeout
            while any(process.poll() is None for process in processes):
                if any(process.poll() not in (None, 0) for process in processes):
                    raise RuntimeError("ERROR reduction worker failed")
                if time.monotonic() >= deadline:
                    raise TimeoutError("ERROR distributed reduction timeout")
                time.sleep(0.1)
            if any(process.returncode != 0 for process in processes):
                raise RuntimeError("ERROR reduction worker failed")
        except Exception as error:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            details: list[str] = []
            for rank, log in enumerate(logs):
                log.flush()
                log.seek(0)
                details.append(f"rank {rank}: {log.read()[-2500:]}")
            raise RuntimeError(f"{error}\n" + "\n".join(details)) from error
        finally:
            for log in logs:
                log.close()
        return [
            (Path(directory) / f"rank-{rank}.log").read_text() for rank in range(count)
        ]


def check_layout(px: int, py: int) -> None:
    """Check replicated sums, halo-free gradients, and directional derivatives."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import Array
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from numpy.typing import NDArray

    from veris.global_sum import global_sum

    nx, ny = 3, 5
    shape = (px * (nx + 4), py * (ny + 4))
    packed: NDArray[np.float64] = np.full(shape, 1e6)
    gradient: NDArray[np.float64] = np.zeros(shape)
    for x in range(px):
        for y in range(py):
            block = (
                slice(x * (nx + 4) + 2, x * (nx + 4) + 2 + nx),
                slice(y * (ny + 4) + 2, y * (ny + 4) + 2 + ny),
            )
            first = (x * py + y) * nx * ny + 1
            interior = np.arange(first, first + nx * ny).reshape(nx, ny)
            packed[block] = interior
            gradient[block] = 2 * interior
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))

    def packed_slice(index: tuple[slice, ...] | None) -> NDArray[np.float64]:
        """Supply the owning rank's slice of the packed global oracle."""
        return packed[index]

    data = jax.make_array_from_callback(shape, sharding, packed_slice)

    def local_norms(value: Array) -> Array:
        """Reduce the local interior sum and squared norm over the mesh."""
        interior = value[2:-2, 2:-2]
        return global_sum(
            jnp.stack([jnp.sum(interior), jnp.sum(interior**2)]),
            axis_names=("x", "y"),
        )

    # shard_map preserves the callback's two-component array output.
    norms = cast(
        Callable[[Array], Array],
        jax.shard_map(local_norms, mesh=mesh, in_specs=P("x", "y"), out_specs=P()),
    )
    n = px * py * nx * ny
    expected: list[float] = [n * (n + 1) / 2, n * (n + 1) * (2 * n + 1) / 6]
    np.testing.assert_allclose(norms(data), expected, rtol=1e-13)

    def tangent_slice(index: tuple[slice, ...] | None) -> NDArray[np.float64]:
        """Supply a unit direction on each rank, including poisoned halo cells."""
        return np.ones(shape)[index]

    def squared_norm(value: Array) -> Array:
        """Select the globally reduced squared norm for reverse differentiation."""
        return norms(value)[1]

    tangent = jax.make_array_from_callback(shape, sharding, tangent_slice)
    with jax.set_mesh(mesh):
        _, directional = jax.jvp(norms, (data,), (tangent,))
        adjoint: Array = jax.grad(squared_norm)(data)
    np.testing.assert_allclose(directional, [n, n * (n + 1)], rtol=1e-13)
    for shard in adjoint.addressable_shards:
        np.testing.assert_allclose(shard.data, gradient[shard.index], rtol=1e-13)


def main() -> None:
    """Initialize before backend access and verify every rank's local gradients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--coordinator")
    args = parser.parse_args()
    import jax

    jax.config.update("jax_enable_x64", True)
    if args.count > 1:
        jax.distributed.initialize(
            coordinator_address=args.coordinator,
            num_processes=args.count,
            process_id=args.rank,
            initialization_timeout=45,
            local_device_ids=[0]
            if os.environ.get("JAX_PLATFORMS") in ("cuda", "gpu")
            else None,
        )
    try:
        assert jax.process_count() == args.count
        n = len(jax.devices())
        layouts = [(n, 1), (1, n)]
        if n == 4:
            layouts.append((2, 2))
        for layout in layouts:
            check_layout(*layout)
        print(
            f"rank {args.rank}: reduction values, gradients, and halo exclusion passed",
            flush=True,
        )
    finally:
        if args.count > 1:
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
