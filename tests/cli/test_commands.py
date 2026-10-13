"""Central commands stay usable without selecting execution resources."""

import importlib
import subprocess
import sys
from pathlib import Path

import jax
import pytest
from click.testing import CliRunner


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--help"],
        ["--version"],
        ["show-setups"],
        ["copy-setup", "--help"],
        ["run", "--help"],
    ],
)
def test_information_commands_do_not_configure_jax(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enumeration during help/discovery is an unwanted execution side effect."""

    def forbid_devices(*args: object, **kwargs: object) -> None:
        raise AssertionError("ERROR information command enumerated devices")

    monkeypatch.setattr(jax, "devices", forbid_devices)
    before = (jax.config.jax_enable_x64, jax.config.jax_platforms)
    command = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(command, arguments)
    assert result.exit_code == 0, result.output
    assert before == (jax.config.jax_enable_x64, jax.config.jax_platforms)
    if arguments in ([], ["--help"]):
        assert all(
            name in result.output for name in ("show-setups", "copy-setup", "run")
        )
    elif arguments == ["--version"]:
        assert "version" in result.output and "Traceback" not in result.output
    elif arguments == ["show-setups"]:
        assert result.output.splitlines() == ["dynamics", "growth", "island"]


def test_module_entry_point_from_source() -> None:
    """The module entry point must expose the same public command group."""
    result = subprocess.run(
        [sys.executable, "-m", "veris.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    assert all(name in result.stdout for name in ("show-setups", "copy-setup", "run"))


def test_copy_command_defaults_to_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cli = importlib.import_module("veris.cli").cli
    result = CliRunner().invoke(cli, ["copy-setup", "growth"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "growth" / "setup.py").is_file()
