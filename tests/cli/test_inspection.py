"""Definition inspection/export is cold, sparse when requested and no-clobber."""

import json
from pathlib import Path
from typing import Any

import jax
import pytest
from click.testing import CliRunner

from veris.cli import cli


@pytest.mark.parametrize("authored", [False, True])
def test_inspect_exports_with_overrides_without_selecting_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authored: bool
) -> None:
    source, destination = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(
        json.dumps(
            {"preset": "growth", "profile": "example", "settings": {"dtype": "float32"}}
        )
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR inspection selected devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    args = [
        "inspect",
        str(source),
        "--export",
        str(destination),
        "--set",
        'settings.dtype="float64"',
    ]
    if authored:
        args.append("--authored")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    data = json.loads(destination.read_text())
    assert data["profile"] == "example" and data["settings"]["dtype"] == "float64"
    if authored:
        assert not data.get("scenario")
    else:
        assert data["scenario"]["hIceMean"] == 1.3


def test_inspection_export_preserves_existing_file(tmp_path: Path) -> None:
    source, destination = tmp_path / "input.json", tmp_path / "existing.json"
    source.write_text(json.dumps({"preset": "growth"}))
    destination.write_text("preserve")
    result = CliRunner().invoke(
        cli, ["inspect", str(source), "--export", str(destination)]
    )
    assert result.exit_code != 0 and "exist" in result.output.lower()
    assert destination.read_text() == "preserve"
