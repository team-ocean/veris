"""Single-device collector semantics shared with distributed netCDF drivers."""

from pathlib import Path

import jax
import numpy as np

from veris import Experiment, Grid, initialize_experiment


def test_collector_removes_halos_for_snapshot_and_preserves_requested_names(
    tmp_path: Path,
) -> None:
    from veris.io import read_record, write_snapshot
    from veris.io.distributed import distributed_collector

    mesh = jax.make_mesh((1, 1), ("x", "y"), devices=[jax.devices("cpu")[0]])
    with jax.set_mesh(mesh):
        state, conf, phys = initialize_experiment(
            Experiment.builtin("dynamics", grid=Grid(3, 4))
        )
        state = jax.device_put(
            state,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", "y")),
        )
        collect = distributed_collector(mesh)
        write_snapshot(
            tmp_path / "physical.nc",
            state,
            variables=("hIceMean",),
            conf=conf,
            phys=phys,
            collector=collect,
        )
    record = read_record(tmp_path / "physical.nc")
    assert record.fields["hIceMean"].shape == (3, 4)
    np.testing.assert_array_equal(
        record.fields["hIceMean"], np.asarray(state.hIceMean)[2:-2, 2:-2]
    )
