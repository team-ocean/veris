"""Setup discovery and copying preserve editable files and reject collisions."""

import importlib
import os
from pathlib import Path

import pytest


def test_discovery_lists_builtin_and_external_templates_without_importing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = tmp_path / "extras"
    setup = extra / "custom"
    setup.mkdir(parents=True)
    (setup / "setup.py").write_text(
        "raise RuntimeError('must not execute during discovery')\n"
    )
    (extra / "unrelated").mkdir()
    monkeypatch.setenv("VERIS_SETUP_DIR", str(extra))
    registry = importlib.import_module("veris.cli.setups")
    discovered = registry.discover_setups()
    assert list(discovered) == ["custom", "dynamics", "growth", "island"]
    assert discovered["custom"] == setup


@pytest.mark.parametrize("name", ["growth", "custom"])
def test_duplicate_setup_names_fail_instead_of_shadowing(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        (root / name).mkdir(parents=True)
        (root / name / "setup.py").write_text("# template\n")
    monkeypatch.setenv("VERIS_SETUP_DIR", os.pathsep.join(map(str, roots)))
    registry = importlib.import_module("veris.cli.setups")
    with pytest.raises(ValueError, match="duplicate"):
        registry.discover_setups()


def test_missing_external_directory_reports_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERIS_SETUP_DIR", str(tmp_path / "missing"))
    registry = importlib.import_module("veris.cli.setups")
    with pytest.raises(ValueError, match="missing"):
        registry.discover_setups()


def test_copy_retains_relative_assets_and_rejects_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "templates"
    source = root / "custom"
    (source / "data").mkdir(parents=True)
    (source / "setup.py").write_text("# custom\n")
    (source / "data" / "mask.txt").write_text("1 0\n")
    monkeypatch.setenv("VERIS_SETUP_DIR", str(root))
    registry = importlib.import_module("veris.cli.setups")
    target = tmp_path / "nested" / "copied"
    assert registry.copy_setup("custom", target) == target
    assert (target / "data" / "mask.txt").read_text() == "1 0\n"
    (target / "setup.py").write_text("# user edits\n")
    with pytest.raises(FileExistsError):
        registry.copy_setup("custom", target)
    assert (target / "setup.py").read_text() == "# user edits\n"


@pytest.mark.parametrize("name", ["unknown", "../growth", "/growth"])
def test_unknown_setup_cannot_create_destination(name: str, tmp_path: Path) -> None:
    registry = importlib.import_module("veris.cli.setups")
    destination = tmp_path / "copied"
    with pytest.raises(ValueError, match="unknown setup"):
        registry.copy_setup(name, destination)
    assert not destination.exists()
