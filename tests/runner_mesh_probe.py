"""Real all-device public runner matches serial spatial forcing and output."""

from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import numpy as np

import veris
from tests.lifecycle_sharding_probe import _custom
from veris.io import OutputSettings, Stream, read_record
from veris.model.types import State


def main() -> None:
    """Acquire four CPU devices through runtime and compare physical records."""
    assert len(jax.devices("cpu")) == 4, "ERROR runner probe expected four devices"
    definition = _custom(
        veris.Grid(8, 12),
        False,
        OutputSettings(
            enabled=False,
            streams=(Stream(sampling_interval=timedelta(seconds=600)),),
        ),
    ).with_overrides({"steps": 2})
    with TemporaryDirectory() as location:
        root = Path(location)
        serial = veris.run_experiment(definition, snapshot_path=root / "serial.nc")
        parallel = veris.run_experiment(
            definition.with_overrides(
                {"execution": {"backend": "cpu", "all_devices": True}}
            ),
            snapshot_path=root / "parallel.nc",
            output_path=root / "history.nc",
        )
        expected, actual = (
            read_record(root / "serial.nc"),
            read_record(root / "parallel.nc"),
        )
        assert serial.elapsed == parallel.elapsed and parallel.completed_steps == 2
        assert parallel.execution_metadata is not None
        assert tuple(parallel.execution_metadata["partitions"]) == (1, 4)
        assert parallel.execution_metadata["process_count"] == 1
        for item in fields(State):
            ref, value = expected.fields[item.name], actual.fields[item.name]
            error = np.abs(value - ref)
            assert np.all(error <= 1e-10 * (1 + np.abs(ref))), (
                f"ERROR runner {item.name}: max_abs={error.max():.6g}"
            )
        assert (
            read_record(root / "history.nc", stream="instantaneous").time
            == parallel.elapsed.total_seconds()
        )
        borrowed = jax.make_mesh((2, 2), ("x", "y"), devices=jax.devices("cpu"))
        veris.run_experiment(
            definition.with_overrides(
                {"execution": {"backend": "cpu", "partitions": (2, 2)}}
            ),
            mesh=borrowed,
            snapshot_path=root / "borrowed.nc",
        )
        saved = read_record(root / "borrowed.nc")
        for item in fields(State):
            ref, value = expected.fields[item.name], saved.fields[item.name]
            error = np.abs(value - ref)
            assert np.all(error <= 1e-10 * (1 + np.abs(ref))), (
                f"ERROR borrowed mesh {item.name}: max_abs={error.max():.6g}"
            )
        sentinel = jax.device_put(
            np.ones((4, 4)),
            jax.sharding.NamedSharding(borrowed, jax.sharding.PartitionSpec("x", "y")),
        )
        with jax.set_mesh(borrowed):
            assert float(sentinel.sum()) == 16.0, "ERROR borrowed mesh became unusable"
    print("runner: four CPU devices match serial fields and scheduled output")


if __name__ == "__main__":
    main()
