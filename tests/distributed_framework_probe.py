"""Real two-process all-mode lifecycle, spatial forcing, histories and ownership."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np

from veris import Experiment, Forcing, Grid, run_experiment
from veris.io import OutputSettings, Stream


def definition(kind: str, cyclic_y: bool) -> Experiment:
    """Build small prescribed inputs crossing the y partition boundary."""
    output = OutputSettings(
        streams=(
            Stream("instant", ("hIceMean", "Area", "uWind"), timedelta(seconds=600)),
            Stream(
                "mean",
                ("hIceMean", "Area", "uWind"),
                timedelta(seconds=600),
                timedelta(seconds=1200),
            ),
        )
    )
    if kind == "custom":
        from tests.lifecycle_sharding_probe import _custom

        model = _custom(Grid(8, 12), cyclic_y, output)
    else:
        model = Experiment.builtin(
            kind,
            grid=Grid(8, 12),
            steps=2,
            settings={
                "deltatDyn": 600.0,
                "deltatTherm": 600.0,
                "nEVPsteps": 2,
                "nITC": 1,
                "enable_cyclic_y": cyclic_y,
            },
            output=output,
        )
    x, y = np.indices((8, 12))
    wind = np.stack([1.0 + 0.01 * x + 0.03 * y, 1.5 + 0.02 * x + 0.04 * y])
    return model.with_overrides(
        {
            "steps": 2,
            "forcing": Forcing(
                constant={"vWind": 1.0}, series={"uWind": wind, "Qnet": [80.0, 100.0]}
            ),
        }
    )


def main() -> None:
    """Run real physics on both ranks; preserve borrowed services and close owned ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("island", "dynamics", "growth", "custom"), required=True
    )
    parser.add_argument("--cyclic-y", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--borrowed", action="store_true")
    parser.add_argument("--local-cpu-devices", type=int, choices=(1, 2), default=1)
    parser.add_argument("--snapshot-failure", action="store_true")
    args = parser.parse_args()
    import jax

    from veris.execution.runtime import create_mesh, distributed_options
    from veris.execution.simulation import Simulation

    rank = int(os.environ.get("JAX_PROCESS_ID", os.environ.get("SLURM_PROCID", "0")))
    partitions = (1, 2 * args.local_cpu_devices if args.borrowed else 2)
    model = definition(args.kind, bool(args.cyclic_y)).with_overrides(
        {"execution": {"backend": "cpu", "partitions": partitions}}
    )
    mesh = None
    if args.borrowed:
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_num_cpu_devices", args.local_cpu_devices)
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
        options = distributed_options(os.environ, "cpu")
        assert options is not None
        jax.distributed.initialize(**options)
        mesh = create_mesh(partitions, "cpu")
    if args.snapshot_failure:

        def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected snapshot failure")

        Simulation.snapshot = fail
    error = None
    try:
        try:
            result = run_experiment(
                model,
                mesh=mesh,
                output_path=args.output_root / "history.nc",
                snapshot_path=args.output_root / "final.nc",
            )
            assert (
                result.completed_steps == 2 and result.elapsed.total_seconds() == 1200.0
            )
            assert (
                result.execution_metadata is not None
                and result.execution_metadata["process_count"] == 2
                and result.execution_metadata["partitions"] == partitions
            )
        except RuntimeError as exc:
            if not args.snapshot_failure or "injected snapshot failure" not in str(exc):
                raise
            error = str(exc)
        assert jax.distributed.is_initialized() is args.borrowed
        if args.borrowed:
            from jax.experimental import multihost_utils

            received = np.asarray(multihost_utils.process_allgather(np.asarray(rank)))
            assert set(received.ravel().tolist()) == {0, 1}, (
                "ERROR borrowed service unusable"
            )
        payload = {
            "rank": rank,
            "kind": args.kind,
            "cyclic_y": bool(args.cyclic_y),
            "borrowed": args.borrowed,
            "closed_owned": not jax.distributed.is_initialized(),
            "error": error,
        }
        (args.output_root / f"rank-{rank}.json").write_text(json.dumps(payload))
    finally:
        if args.borrowed:
            jax.distributed.shutdown()


if __name__ == "__main__":
    main()
