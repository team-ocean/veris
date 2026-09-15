"""Exercise real model drivers, AD and partition-aware netCDF output."""

import os
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("driver", ["run_growth", "run_dyn"])
def test_serial_driver_writes_selected_stream_and_final_snapshot(
    tmp_path: Path, driver: str
) -> None:
    from importlib import import_module

    from veris.io import read_record

    module = import_module(f"veris.setups.{driver}")
    interval = "86400" if driver == "run_growth" else "600"
    extra = (
        [] if driver == "run_growth" else ["--nx", "3", "--ny", "4", "--evp-steps", "2"]
    )
    history, final = (tmp_path / n for n in ("history.nc", "final.nc"))
    module.main(
        [
            "--steps",
            "1",
            "--output",
            str(final),
            "--netcdf",
            str(history),
            "--io-variables",
            "hIceMean,Area",
            "--sample-seconds",
            interval,
            "--average",
            "instantaneous,daily",
            *extra,
        ]
    )
    sampled = read_record(history, stream="instantaneous")
    snapshot = read_record(final)
    assert set(sampled.fields) == {"hIceMean", "Area"}
    assert sampled.time == float(interval) and snapshot.time == float(interval)
    np.testing.assert_array_equal(
        sampled.fields["hIceMean"], snapshot.fields["hIceMean"]
    )
    expected_shape = (2, 2) if driver == "run_growth" else (3, 4)
    assert snapshot.fields["hIceMean"].shape == expected_shape


def test_real_growth_ad_disables_output_and_writes_auxiliary_final_state(
    tmp_path: Path,
) -> None:
    from veris.io import OutputManager, OutputSettings, read_record, write_snapshot
    from veris.setups.run_growth import compiled_step, initialize

    state, conf, phys = initialize()
    forbidden = tmp_path / "during_ad.nc"
    with OutputManager(forbidden, OutputSettings(enabled=False)) as output:

        def objective(longwave: jax.Array) -> tuple[jax.Array, object]:
            result = replace(state, LWdown=jnp.full_like(state.LWdown, longwave))
            for step in range(2):
                result = compiled_step(result, conf, phys)
                output.sample(result, timedelta(days=step + 1))
            return result.hIceMean[2:-2, 2:-2].mean(), result

        (_, final), gradient = jax.value_and_grad(objective, has_aux=True)(80.0)
    assert np.isfinite(gradient) and abs(gradient) > 1e-8
    assert not forbidden.exists()
    write_snapshot(
        tmp_path / "after_ad.nc", final, conf=conf, phys=phys, elapsed=timedelta(days=2)
    )
    np.testing.assert_array_equal(
        read_record(tmp_path / "after_ad.nc").fields["hIceMean"],
        final.hIceMean[2:-2, 2:-2],
    )


def test_parallel_driver_gathers_each_partition_before_netcdf(tmp_path: Path) -> None:
    from veris.io import read_record

    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", PYTHONPATH=str(root)
    )
    final, history = (tmp_path / n for n in ("final.nc", "history.nc"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "veris.setups.run_parallel",
            "--nx",
            "8",
            "--ny",
            "8",
            "--mesh",
            "2",
            "2",
            "--steps",
            "1",
            "--evp-steps",
            "2",
            "--output",
            str(final),
            "--netcdf",
            str(history),
            "--io-variables",
            "hIceMean,uIce",
            "--sample-seconds",
            "600",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR netCDF parallel driver: {result.stderr[-1800:]}"
    )
    snapshot = read_record(final)
    sampled = read_record(history, stream="instantaneous")
    for name in ("hIceMean", "uIce"):
        assert snapshot.fields[name].shape == (8, 8)
        np.testing.assert_array_equal(sampled.fields[name], snapshot.fields[name])


def test_driver_rejects_unreachable_sampling_time_before_creating_output(
    tmp_path: Path,
) -> None:
    from veris.setups import run_growth

    with pytest.raises(ValueError, match="multiple"):
        run_growth.main(
            [
                "--steps",
                "1",
                "--output",
                str(tmp_path / "final.nc"),
                "--netcdf",
                str(tmp_path / "bad.nc"),
                "--sample-seconds",
                "1",
            ]
        )
    assert not (tmp_path / "bad.nc").exists()
