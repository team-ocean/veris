"""Single-device collector semantics shared with distributed netCDF drivers."""

from pathlib import Path

import jax
import numpy as np


def test_collector_removes_halos_for_snapshot_and_preserves_requested_names(
    tmp_path: Path,
) -> None:
    from veris.io import read_record, write_snapshot
    from veris.io.distributed import distributed_collector
    from veris.setups import run_dyn

    mesh = jax.make_mesh((1, 1), ("x", "y"), devices=[jax.devices("cpu")[0]])
    with jax.set_mesh(mesh):
        state, conf, phys = run_dyn.initialize(3, 4, mesh=mesh)
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
