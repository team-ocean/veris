"""Public runner preserves real physics, counters, output and resource cleanup."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from importlib import import_module
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

import veris
from tests.lifecycle_sharding_probe import _custom
from tests.process_helpers import wait_for_processes
from veris.io import OutputSettings, read_record
from veris.model.types import State


def test_public_runner_matches_real_simulation_all_fields_and_diagnostics(
    tmp_path: Path,
) -> None:
    # An unspecified backend honors the caller's JAX choice, including GPUs.
    expected_backend = jax.default_backend()
    definition = _custom(
        veris.Grid(8, 12), True, OutputSettings(enabled=False)
    ).with_overrides({"steps": 2})
    with veris.Simulation.initialize(definition) as simulation:
        expected = simulation.run()
    destination = tmp_path / "nested" / "final.nc"
    actual = veris.run_experiment(definition, snapshot_path=destination)
    assert isinstance(actual, veris.RunResult)
    assert actual.steps == actual.completed_steps == expected.completed_steps == 2
    assert actual.elapsed == expected.elapsed
    for item in fields(State):
        np.testing.assert_array_equal(
            getattr(actual.state, item.name),
            getattr(expected.state, item.name),
            err_msg=f"ERROR runner {item.name}",
        )
    assert actual.diagnostics is not None and expected.diagnostics is not None
    for item in fields(actual.diagnostics):
        np.testing.assert_array_equal(
            getattr(actual.diagnostics, item.name),
            getattr(expected.diagnostics, item.name),
            err_msg=f"ERROR diagnostics {item.name}",
        )
    saved = read_record(destination)
    assert set(saved.fields) == {item.name for item in fields(State)}
    assert saved.time == actual.elapsed.total_seconds()
    assert actual.execution_metadata is not None
    assert actual.execution_metadata["backend"] == expected_backend
    assert tuple(actual.execution_metadata["partitions"]) == (1, 1)


def test_runner_zero_steps_and_override_precedence(tmp_path: Path) -> None:
    definition = veris.Experiment.builtin("growth", steps=3)
    path = tmp_path / "initial.nc"
    result = veris.run_experiment(
        definition,
        overrides={"steps": 0, "initial_conditions": {"hIceMean": 0.4}},
        snapshot_path=path,
        variables=("hIceMean", "Area"),
    )
    assert result.steps == result.completed_steps == 0 and result.diagnostics is None
    assert result.elapsed.total_seconds() == 0
    saved = read_record(path)
    assert set(saved.fields) == {"hIceMean", "Area"}
    np.testing.assert_array_equal(saved.fields["hIceMean"], 0.4)
    assert definition.steps == 3 and not definition.initial_conditions


@pytest.mark.parametrize("failure", ["run", "snapshot"])
def test_runner_releases_resources_when_execution_or_snapshot_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    runner = import_module("veris.execution.runner")
    runtime = import_module("veris.execution.runtime")
    real = runtime.execution_resources
    events = []

    @contextmanager
    def resources(*args: Any, **kwargs: Any) -> Iterator[Any]:
        with real(*args, **kwargs) as selected:
            events.append("enter")
            try:
                yield selected
            finally:
                events.append("exit")

    def failed(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"{failure} failed")

    monkeypatch.setattr(runner, "execution_resources", resources)
    monkeypatch.setattr(veris.Simulation, failure, failed)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        veris.run_experiment(
            veris.Experiment.builtin("growth", steps=0),
            snapshot_path=tmp_path / "failed.nc",
        )
    assert events == ["enter", "exit"]
    assert not (tmp_path / "failed.nc").exists()


@pytest.mark.parametrize(
    "bad", ["precision", "variables", "existing", "collision", "disabled_path"]
)
def test_runner_invalid_preflight_has_no_device_or_directory_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: str
) -> None:
    definition = veris.Experiment.builtin("growth", steps=0)
    output = tmp_path / "new" / "history.nc"
    snapshot = tmp_path / "new" / "snapshot.nc"
    options: dict[str, Any] = {"snapshot_path": snapshot}
    if bad == "precision":
        definition = definition.with_overrides(
            {"settings": {"dtype": "float32"}, "forcing": {"constant": {"Qnet": 1e40}}}
        )
    elif bad == "variables":
        options["variables"] = ("unknown",)
    elif bad == "existing":
        snapshot.parent.mkdir()
        snapshot.write_text("preserve")
    elif bad == "collision":
        options.update(output_path=output, snapshot_path=output)
    else:
        definition = definition.with_overrides({"output": {"enabled": True}})

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR invalid runner preflight selected devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    monkeypatch.setattr(jax.distributed, "initialize", forbidden)
    with pytest.raises((ValueError, FileExistsError)):
        veris.run_experiment(definition, **options)
    if bad == "existing":
        assert snapshot.read_text() == "preserve"
    else:
        assert not output.parent.exists()


def test_runner_requested_history_uses_shared_default_schedule(tmp_path: Path) -> None:
    definition = veris.Experiment.builtin(
        "growth", steps=2, settings={"deltatTherm": 60.0, "nITC": 1}
    )
    history = tmp_path / "history" / "run.nc"
    result = veris.run_experiment(
        definition, output_path=history, variables=("hIceMean", "Area")
    )
    saved = read_record(history, stream="instantaneous")
    assert saved.time == result.elapsed.total_seconds() == 120.0
    assert set(saved.fields) == {"hIceMean", "Area"}


def test_real_public_runner_on_four_local_cpu_devices() -> None:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", PYTHONPATH=str(root)
    )
    for name in (
        "JAX_COORDINATOR_ADDRESS",
        "JAX_NUM_PROCESSES",
        "JAX_PROCESS_ID",
        "SLURM_PROCID",
        "SLURM_LOCALID",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, str(root / "tests/runner_mesh_probe.py")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"ERROR public runner mesh: {result.stderr[-2500:]}"
    assert "four CPU devices match serial fields and scheduled output" in result.stdout


@pytest.mark.parametrize("steps", [0, 1])
def test_real_runner_owns_two_local_processes_and_preserves_outputs(
    tmp_path: Path, steps: int
) -> None:
    import json
    import os
    import socket
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    history, snapshot = tmp_path / "history.nc", tmp_path / "final.nc"
    code = """
import json, sys
from pathlib import Path
import jax
from veris import Experiment, Grid, Execution, run_experiment
result = run_experiment(
    Experiment.builtin("dynamics", grid=Grid(4, 8), steps=int(sys.argv[3]),
        settings={"nEVPsteps": 2}, execution=Execution(backend="cpu", partitions=(1, 2))),
    output_path=Path(sys.argv[1]), snapshot_path=Path(sys.argv[2]),
)
assert not jax.distributed.is_initialized(), "ERROR owned service remains initialized"
print(json.dumps(dict(result.execution_metadata)), flush=True)
"""
    processes = []
    logs = []
    try:
        for rank in range(2):
            environment = dict(
                os.environ,
                JAX_PLATFORMS="cpu",
                JAX_NUM_CPU_DEVICES="1",
                PYTHONPATH=str(root),
                JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}",
                JAX_NUM_PROCESSES="2",
                JAX_PROCESS_ID=str(rank),
            )
            for name in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"):
                environment.pop(name, None)
            log = tmp_path / f"runner-rank-{rank}.log"
            logs.append(log)
            with log.open("w") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        code,
                        str(history),
                        str(snapshot),
                        str(steps),
                    ],
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append(process)
        try:
            wait_for_processes(processes)
        except ChildProcessError as error:
            details = "\n".join(log.read_text()[-2500:] for log in logs)
            raise AssertionError(f"{error}\n{details}") from error
        for rank, process in enumerate(processes):
            stdout = stderr = logs[rank].read_text()
            assert process.returncode == 0, f"ERROR runner rank{rank}: {stderr[-2500:]}"
            metadata = json.loads(stdout.strip().splitlines()[-1])
            assert metadata["process_count"] == 2 and metadata["process_index"] == rank
            assert metadata["partitions"] == [1, 2]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
    actual = read_record(snapshot)
    with veris.Simulation.initialize(
        veris.Experiment.builtin(
            "dynamics", grid=veris.Grid(4, 8), steps=steps, settings={"nEVPsteps": 2}
        )
    ) as model:
        expected = model.run()
    for item in fields(State):
        ref = np.asarray(getattr(expected.state, item.name))[2:-2, 2:-2]
        value = actual.fields[item.name]
        error = np.abs(value - ref)
        assert np.all(error <= 1e-10 * (1 + np.abs(ref))), (
            f"ERROR two-process {item.name}: max_abs={error.max():.6g}"
        )
    assert (
        read_record(history, stream="instantaneous").time
        == actual.time
        == expected.elapsed.total_seconds()
    )


def test_runner_records_actual_execution_in_snapshot_and_history(
    tmp_path: Path,
) -> None:
    import json

    import h5netcdf

    definition = veris.Experiment.builtin("growth", steps=0)
    history, snapshot = tmp_path / "history.nc", tmp_path / "final.nc"
    result = veris.run_experiment(
        definition, output_path=history, snapshot_path=snapshot
    )
    assert result.execution_metadata is not None
    expected = dict(result.execution_metadata)
    expected["partitions"] = list(expected["partitions"])
    for path in (history, snapshot):
        with h5netcdf.File(path, "r") as file:
            assert json.loads(file.attrs["execution"]) == expected


def test_requested_history_preserves_an_authored_sampling_interval(
    tmp_path: Path,
) -> None:
    from datetime import timedelta

    from veris.io import Stream

    definition = veris.Experiment.builtin(
        "growth",
        steps=2,
        settings={"deltatTherm": 600.0},
        output=OutputSettings(
            enabled=False, streams=(Stream(sampling_interval=timedelta(hours=1)),)
        ),
    )
    history = tmp_path / "history.nc"
    result = veris.run_experiment(definition, output_path=history)
    assert result.elapsed.total_seconds() == 1200.0
    assert read_record(history, stream="instantaneous").time == 0.0
    assert definition.output is not None and not definition.output.enabled
