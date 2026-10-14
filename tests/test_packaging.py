"""Built source distributions retain the complete scientific verification harness."""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    (
        ("veris.model.configuration", "Configuration"),
        ("veris.model.physical_constants", "PhysicalConstants"),
        ("veris.setups.island", "IslandSettings"),
        ("veris.setups.run_growth", "GrowthSettings"),
    ),
)
def test_documented_registry_classes_have_viewable_source(
    module_name: str, class_name: str
) -> None:
    """Registry decorators must allow Sphinx to generate real source links."""
    from sphinx.errors import PycodeError
    from sphinx.pycode import ModuleAnalyzer

    try:
        analyzer = ModuleAnalyzer.for_module(module_name)
        analyzer.find_tags()
    except PycodeError as error:
        pytest.fail(f"ERROR documentation source {module_name}: {error}", pytrace=False)
    assert class_name in analyzer.tags, (
        f"ERROR documentation source lacks {module_name}.{class_name}"
    )


def test_sdist_contains_test_helpers_references_and_documented_architecture(
    tmp_path: Path,
) -> None:
    """Build in a copy so packaging cannot mutate the running source checkout."""
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("veris", "tests", "docs", "benchmarks"):
        shutil.copytree(
            root / name,
            source / name,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "_build"),
        )
    for name in (
        "setup.py",
        "setup.cfg",
        "versioneer.py",
        "versioneer.pyi",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "pytest.ini",
        "requirements-test.txt",
        "requirements-profile.txt",
        "DESIGN.md",
        "CHANGELOG.md",
    ):
        shutil.copyfile(root / name, source / name)
    result = subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build",
            "--build-base",
            str(tmp_path / "build"),
            "bdist_wheel",
            "--dist-dir",
            str(tmp_path / "dist"),
            "sdist",
            "--dist-dir",
            str(tmp_path / "dist"),
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"ERROR isolated sdist build: {result.stderr[-1800:]}"
    )
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as archive:
        entries = {
            "/".join(Path(member.name).parts[1:]) for member in archive.getmembers()
        }
    required = {
        "pytest.ini",
        "requirements-test.txt",
        "DESIGN.md",
        "CHANGELOG.md",
        "veris/setups/run_parallel.slurm",
    }
    for path in (source / "tests").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".json", ".npz", ".txt"}:
            required.add(str(path.relative_to(source)))
    for path in (source / "docs").rglob("*"):
        if path.is_file() and path.suffix in {
            ".md",
            ".rst",
            ".py",
            ".json",
            ".txt",
            ".conf",
        }:
            required.add(str(path.relative_to(source)))
    # The archive must run its benchmark tests and retain source documentation.
    required.add("requirements-profile.txt")
    for path in (root / "benchmarks").rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md"}:
            required.add(str(path.relative_to(root)))
    for path in (source / "veris").rglob("*.py"):
        required.add(str(path.relative_to(source)))
    missing = sorted(required - entries)
    assert not missing, (
        f"ERROR sdist omits {len(missing)} harness/docs files: {missing[:8]}"
    )
    installed = tmp_path / "installed"
    installation = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--target",
            str(installed),
            str(next((tmp_path / "dist").glob("*.whl"))),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert installation.returncode == 0, (
        f"ERROR wheel install: {installation.stderr[-1500:]}"
    )
    command = installed / "bin" / "veris"
    assert command.is_file(), "ERROR installed wheel has no veris console script"
    environment = dict(os.environ, PYTHONPATH=str(installed), JAX_PLATFORMS="cpu")
    for arguments in (
        ["--version"],
        ["show-setups"],
        ["copy-setup", "growth", "--to", str(tmp_path / "copied")],
        [
            "run",
            str(tmp_path / "copied" / "setup.py"),
            "--steps",
            "1",
            "--float-type",
            "float32",
            "--output",
            str(tmp_path / "final.nc"),
        ],
    ):
        execution = subprocess.run(
            [str(command), *arguments],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert execution.returncode == 0, (
            f"ERROR installed CLI {arguments[0]}: {execution.stderr[-1500:]}"
        )
    from veris.io import read_record

    saved = read_record(tmp_path / "final.nc")
    assert saved.time == 86400 and saved.fields["hIceMean"].shape == (2, 2)
    assert str(saved.fields["hIceMean"].dtype) == "float32"
