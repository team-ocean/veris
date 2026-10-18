"""Real two-process all-mode lifecycle, spatial forcing, histories and ownership."""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np

from veris import Execution, Experiment, Forcing, Grid, run_experiment
from veris.io import OutputSettings, Stream

DYNAMICS_REPETITIONS = 32


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


def repeated_dynamics_definition() -> Experiment:
    """Use the strong reference wind and nested EVP loop at a real halo cut."""
    return Experiment.builtin(
        "dynamics",
        grid=Grid(64, 64),
        steps=2,
        settings={"nEVPsteps": 4},
        execution=Execution(backend="cpu"),
    )


def repeat_dynamics(output_root: Path, partitions: tuple[int, int]) -> None:
    """Compare fresh distributed rollouts with all serial physical fields.

    The native two-node failure is intermittent: independent final halo
    exchanges can overlap the nested EVP loop. Recompile each trajectory and
    inspect addressable interiors without adding collective synchronization
    between the solver and its halo exchanges. The parent writes the serial
    reference before either rank initializes its backend.
    """
    import jax

    from veris import Simulation, initialize_experiment
    from veris.execution.runtime import create_mesh, distributed_options
    from veris.grid.regions import physical_indices
    from veris.io import read_record

    reference = read_record(output_root / "reference.nc").fields
    assert len(reference) == 70, "ERROR repeated dynamics reference field count"
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    jax.config.update("jax_num_cpu_devices", 1)
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    options = distributed_options(os.environ, "cpu")
    assert options is not None, "ERROR repeated dynamics needs two processes"
    jax.distributed.initialize(**options)
    try:
        mesh = create_mesh(partitions, "cpu")
        model = repeated_dynamics_definition().with_overrides(
            {"execution": {"partitions": partitions}}
        )
        initial, conf, phys = initialize_experiment(model, mesh=mesh)
        failures = []
        for trial in range(DYNAMICS_REPETITIONS):
            with (
                jax.default_device(mesh.local_devices[0]),
                Simulation(model, initial, conf, phys, mesh) as simulation,
            ):
                result = simulation.run()
                jax.block_until_ready(result.state)
                assert result.elapsed.total_seconds() == 1200.0
                for name, expected in reference.items():
                    for shard in getattr(result.state, name).addressable_shards:
                        index = tuple(
                            slice(part.start or 0, part.stop or extent)
                            for part, extent in zip(
                                shard.index,
                                model.grid.storage_shape(partitions),
                                strict=True,
                            )
                        )
                        x, y = physical_indices(index, conf, (64, 64))
                        target = expected[np.ix_(x[2:-2], y[2:-2])]
                        actual = np.asarray(shard.data)[2:-2, 2:-2]
                        scaled = float(
                            np.max(np.abs(actual - target) / (1 + np.abs(target)))
                        )
                        if not scaled <= 1e-10:
                            failures.append((trial, name, scaled))
            jax.clear_caches()
        assert not failures, (
            f"ERROR repeated dynamics rank{jax.process_index()}: "
            f"{len(failures)} mismatches, first={failures[:3]}"
        )
        print(
            f"PASS repeated dynamics rank{jax.process_index()}: "
            f"{DYNAMICS_REPETITIONS} trajectories, 70 fields"
        )
    finally:
        jax.distributed.shutdown()


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
    parser.add_argument("--repeat-dynamics", action="store_true")
    parser.add_argument("--dynamics-mesh", nargs=2, type=int, default=(1, 2))
    args = parser.parse_args()
    if args.repeat_dynamics:
        repeat_dynamics(args.output_root, tuple(args.dynamics_mesh))
        return
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
            assert result.diagnostics is not None
            assert len(jax.tree.leaves(model.physical_constants)) > 1
            for name in (
                "physics_status",
                "ocean_mass_flux",
                "ocean_salt_mass_flux",
                "external_mass_flux",
                "external_heat_flux",
            ):
                for shard in getattr(result.diagnostics, name).addressable_shards:
                    assert np.isfinite(np.asarray(shard.data)).all(), (
                        f"ERROR nonfinite distributed {name}"
                    )
                    if name == "physics_status":
                        assert np.all(np.asarray(shard.data) == 0), (
                            "ERROR distributed physics status"
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
