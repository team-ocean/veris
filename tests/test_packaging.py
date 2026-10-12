"""Built source distributions retain the complete scientific verification harness."""

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


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
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(tmp_path / "dist")],
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
    for path in (source / "docs").rglob("*.md"):
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
