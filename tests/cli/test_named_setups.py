"""Named external setups share discovery, file loading and model execution."""

import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from click.testing import CliRunner

from veris import Simulation
from veris.cli import cli
from veris.cli.run import load_experiment
from veris.io import read_record


@pytest.fixture
def custom_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Author a named setup outside the checkout with relative input assets."""
    root = tmp_path / "external"
    source = root / "my-column"
    copied = CliRunner().invoke(cli, ["copy-setup", "growth", "--to", str(source)])
    assert copied.exit_code == 0, copied.output
    (source / "thickness.txt").write_text("2.5\n")
    (source / "setup.py").write_text(
        "from pathlib import Path\n"
        "from veris import Experiment\n"
        "def build_experiment():\n"
        "    thickness = float((Path(__file__).parent / 'thickness.txt').read_text())\n"
        "    return Experiment.builtin('growth', steps=2, "
        "scenario={'hIceMean': thickness})\n"
    )
    monkeypatch.setenv("VERIS_SETUP_DIR", str(root))
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    return source


def test_named_external_setup_can_be_discovered_copied_and_inspected(
    custom_setup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chosen name and edited definition survive the entire cold workflow."""

    def forbid_devices(*args: object, **kwargs: object) -> None:
        pytest.fail("ERROR named inspection selected devices")

    monkeypatch.setattr(jax, "devices", forbid_devices)
    configuration = (jax.config.jax_enable_x64, jax.config.jax_platforms)
    listed = CliRunner().invoke(cli, ["show-setups"])
    assert listed.exit_code == 0, listed.output
    assert "my-column" in listed.output.splitlines()
    target = tmp_path / "copied"
    copied = CliRunner().invoke(cli, ["copy-setup", "my-column", "--to", str(target)])
    assert copied.exit_code == 0, copied.output
    assert (target / "thickness.txt").read_text() == "2.5\n"
    assert (target / "setup.py").read_bytes() == (
        custom_setup / "setup.py"
    ).read_bytes()
    inspected = CliRunner().invoke(cli, ["inspect", "my-column", "--set", "steps=1"])
    assert inspected.exit_code == 0, inspected.output
    definition = json.loads(inspected.output)
    assert definition["scenario"]["hIceMean"] == 2.5
    assert definition["steps"] == 1
    assert configuration == (jax.config.jax_enable_x64, jax.config.jax_platforms)


def test_named_external_run_matches_python_with_assets_and_overrides(
    custom_setup: Path,
) -> None:
    """Named CLI execution must run the edited factory through the shared lifecycle."""
    working, import_path = Path.cwd(), list(sys.path)
    result = CliRunner().invoke(
        cli,
        [
            "run",
            "my-column",
            "--steps",
            "1",
            "--backend",
            "cpu",
            "--float-type",
            "float32",
            "--output",
            "final.nc",
        ],
    )
    assert result.exit_code == 0, result.output
    assert Path.cwd() == working and sys.path == import_path
    saved = read_record(working / "final.nc")
    assert saved.time == 86400
    assert saved.fields["hIceMean"].dtype == np.dtype("float32")
    definition = load_experiment(custom_setup / "setup.py").with_overrides(
        {
            "steps": 1,
            "settings": {"dtype": "float32"},
            "constants": {"dtype": "float32"},
            "execution": {"backend": "cpu"},
        }
    )
    with Simulation.initialize(definition) as model:
        state = model.run().state
    for name, actual in saved.fields.items():
        expected = np.asarray(getattr(state, name))[2:-2, 2:-2]
        np.testing.assert_array_equal(
            actual, expected, err_msg=f"ERROR named setup {name}"
        )


@pytest.mark.parametrize("command", ["run", "inspect"])
def test_builtin_name_is_runnable_and_inspectable(command: str, tmp_path: Path) -> None:
    """Builtin names use the same registry and overrides as external names."""
    arguments = [command, "growth", "--set", "steps=0"]
    if command == "run":
        arguments.extend(["--output", str(tmp_path / "final.nc")])
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 0, result.output
    if command == "run":
        saved = read_record(tmp_path / "final.nc")
        assert saved.time == 0
        np.testing.assert_array_equal(saved.fields["hIceMean"], np.full((2, 2), 1.3))
    else:
        assert json.loads(result.output)["steps"] == 0


@pytest.mark.parametrize("command", ["run", "inspect"])
@pytest.mark.parametrize("problem", ["unknown", "duplicate", "missing-root"])
def test_invalid_named_setup_fails_before_factory_or_output(
    command: str, problem: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolution failures must be informative and have no execution side effects."""
    root = tmp_path / "templates"
    root.mkdir()
    for name in ("growth",) if problem == "duplicate" else ("my-column",):
        folder = root / name
        folder.mkdir()
        (folder / "setup.py").write_text("raise AssertionError('factory executed')\n")
    monkeypatch.setenv(
        "VERIS_SETUP_DIR", str(root / "absent" if problem == "missing-root" else root)
    )
    name = "unregistered" if problem == "unknown" else "growth"
    destination = tmp_path / "uncreated" / "output.nc"
    arguments = [command, name]
    if command == "run":
        arguments.extend(["--output", str(destination)])
    else:
        arguments.extend(["--export", str(destination)])
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code != 0 and "Error:" in result.output
    reason = {
        "unknown": "unknown setup",
        "duplicate": "duplicate",
        "missing-root": "absent",
    }[problem]
    assert reason in result.output, result.output
    assert "Traceback" not in result.output and "factory executed" not in result.output
    assert not destination.parent.exists()


@pytest.mark.parametrize("command", ["run", "inspect"])
def test_explicit_file_does_not_require_valid_template_roots(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding name lookup must not subject explicit files to unrelated registry errors."""
    source = tmp_path / "explicit.json"
    source.write_text('{"preset": "growth", "steps": 0}')
    monkeypatch.setenv("VERIS_SETUP_DIR", str(tmp_path / "absent"))
    arguments = [command, str(source)]
    if command == "run":
        arguments.extend(["--output", str(tmp_path / "final.nc")])
    result = CliRunner().invoke(cli, arguments)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", ["run", "inspect"])
@pytest.mark.parametrize("local_entry", ["file", "directory", "dangling-symlink"])
def test_existing_file_takes_precedence_over_registered_name(
    command: str, local_entry: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local filesystem entry must not unexpectedly execute a named factory."""
    monkeypatch.chdir(tmp_path)
    if local_entry == "file":
        Path("growth").write_text('{"preset": "growth", "steps": 0}')
    elif local_entry == "directory":
        Path("growth").mkdir()
    else:
        Path("growth").symlink_to(tmp_path / "absent")
    result = CliRunner().invoke(cli, [command, "growth"])
    reason = "extension" if local_entry == "file" else "not a file"
    assert result.exit_code != 0 and reason in result.output, result.output
    assert not Path("output.nc").exists()


@pytest.mark.parametrize("command", ["run", "inspect"])
def test_missing_explicit_path_does_not_fall_back_to_a_builtin(
    command: str, tmp_path: Path
) -> None:
    """Missing paths retain their context and cannot silently select a builtin."""
    source = tmp_path / "missing" / "growth"
    result = CliRunner().invoke(cli, [command, str(source)])
    assert result.exit_code != 0 and str(source) in result.output, result.output


@pytest.mark.parametrize("command", ["run", "inspect"])
@pytest.mark.parametrize("source", ["./growth", "growth.py", "growth.json"])
def test_missing_relative_path_does_not_run_registered_name(
    command: str, source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ./ prefix must survive argument parsing and prevent name lookup."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, [command, source, "--set", "steps=0"])
    assert result.exit_code != 0 and "not a file" in result.output, result.output
    assert not Path("output.nc").exists()
