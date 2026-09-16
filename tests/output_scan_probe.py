"""Real device-sharded scheduled means against serial and arithmetic oracles.

Run with JAX_NUM_CPU_DEVICES=4 python tests/output_scan_probe.py --backend cpu;
use --backend gpu for a required two-device accelerator check. Halo-filled tile
storage is deliberately distinct from global physical order. Small additive
fields isolate schedule, reduction precision, sharding and collector behavior.
"""

import argparse
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from veris.integration_output import output_callbacks, run_timed
from veris.io import OutputManager, OutputSettings, Stream
from veris.io.distributed import distributed_collector
from veris.setups.run_parallel import remove_halos


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    """Report concise aggregate evidence instead of printing entire fields."""
    left, right = np.asarray(actual), np.asarray(expected)
    assert left.shape == right.shape, (
        f"ERROR {label}: shape {left.shape} != {right.shape}"
    )
    assert np.isfinite(left).all(), f"ERROR {label}: nonfinite values"
    error = np.max(np.abs(left - right))
    assert error == 0, f"ERROR {label}: max absolute difference {error}"


def check(backend: str, dtype: str, directory: Path) -> None:
    """Require actual devices and compare every physical output record."""
    devices = jax.devices(backend)
    required = 4 if backend == "cpu" else 2
    assert len(devices) >= required, (
        f"ERROR require {required} {backend} devices, got {len(devices)}"
    )
    px, py = 2, required // 2
    mesh = jax.make_mesh((px, py), ("x", "y"), devices=devices[:required])
    physical = np.arange(6 * 4 * py, dtype=dtype).reshape(6, 4 * py)
    tiles = [
        [
            np.pad(
                physical[x * 3 : (x + 1) * 3, y * 4 : (y + 1) * 4],
                2,
                constant_values=-999,
            )
            for y in range(py)
        ]
        for x in range(px)
    ]
    storage = np.block(tiles)
    settings = OutputSettings(
        streams=(
            Stream("four", ("Area",), timedelta(seconds=1), timedelta(seconds=4)),
            Stream("six", ("Area",), timedelta(seconds=2), timedelta(seconds=6)),
            Stream("instant", ("Area",), timedelta(seconds=5)),
        )
    )

    def advance(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {"Area": state["Area"] + jnp.asarray(1, dtype=state["Area"].dtype)}

    paths = {
        kind: directory / f"{backend}-{dtype}-{kind}.nc"
        for kind in ("serial", "parallel")
    }
    with jax.enable_x64(dtype == "float64"):
        serial = {"Area": jnp.asarray(np.pad(physical, 2, constant_values=-999))}
        with OutputManager(paths["serial"], settings) as manager:
            observe, select = output_callbacks(manager, 1)
            serial_final, _, _ = run_timed(
                serial, advance, 13, observe=observe, select=select
            )
        calls = []
        gather = distributed_collector(mesh)

        def collect(fields: Any, names: tuple[str, ...]) -> Any:
            assert names == ("Area",)
            array = fields["Area"]
            assert array.shape == storage.shape, (
                "ERROR collector received already-trimmed array"
            )
            assert array.sharding.num_devices == required, (
                "ERROR reduced array lost device sharding"
            )
            calls.append(array.dtype)
            return gather(fields, names)

        with jax.set_mesh(mesh):
            parallel = {
                "Area": jax.device_put(storage, NamedSharding(mesh, P("x", "y")))
            }
            with OutputManager(
                paths["parallel"], settings, collector=collect
            ) as manager:
                observe, select = output_callbacks(manager, 1)
                parallel_final, _, _ = run_timed(
                    parallel, advance, 13, observe=observe, select=select
                )
            final_physical = remove_halos(parallel_final["Area"], mesh)
        assert jax.config.x64_enabled == (dtype == "float64")
        _assert_equal(final_physical, physical + 13, "final arithmetic oracle")
        _assert_equal(final_physical, serial_final["Area"][2:-2, 2:-2], "serial final")
        assert parallel_final["Area"].dtype == np.dtype(dtype)
    assert len(calls) == 8, f"ERROR expected 8 record collections, got {len(calls)}"
    assert sum(value == np.dtype("float64") for value in calls) >= 5
    offsets = {"four": [1.5, 5.5, 9.5], "six": [2, 8], "instant": [0, 5, 10]}
    counts = {"four": [4, 4, 4], "six": [3, 3], "instant": [1, 1, 1]}
    bounds = {
        "four": [[0, 4], [4, 8], [8, 12]],
        "six": [[0, 6], [6, 12]],
        "instant": [[0, 0], [5, 5], [10, 10]],
    }
    with (
        h5netcdf.File(paths["serial"]) as serial_file,
        h5netcdf.File(paths["parallel"]) as parallel_file,
    ):
        assert set(parallel_file.groups) == set(offsets)
        for name, means in offsets.items():
            actual, expected = parallel_file.groups[name], serial_file.groups[name]
            for variable in expected.variables:
                _assert_equal(
                    actual.variables[variable][:],
                    expected.variables[variable][:],
                    f"{name}/{variable} serial",
                )
            _assert_equal(
                actual.variables["Area"][:],
                np.stack([physical + mean for mean in means]),
                f"{name} arithmetic oracle",
            )
            _assert_equal(
                actual.variables["sample_count"][:], counts[name], f"{name} counts"
            )
            _assert_equal(
                actual.variables["time_bounds"][:], bounds[name], f"{name} bounds"
            )


def main() -> None:
    """Run both physics precisions on the explicitly required backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    arguments = parser.parse_args()
    with TemporaryDirectory(prefix="veris-output-scan-") as directory:
        for dtype in ("float32", "float64"):
            check(arguments.backend, dtype, Path(directory))
    print(
        f"scheduled output passed on {arguments.backend}: float32/float64, actual collectors, 8 records each"
    )


if __name__ == "__main__":
    main()
