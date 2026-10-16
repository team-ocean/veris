"""Documentation checkouts must retain the Git metadata used by Versioneer."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("shallow", [False, True])
@pytest.mark.parametrize("distance", [0, 2])
def test_readthedocs_checkout_resolves_tagged_and_development_versions(
    tmp_path: Path, shallow: bool, distance: int
) -> None:
    """Run the configured checkout hooks against a real clone without tags."""
    root = Path(__file__).resolve().parents[1]
    origin = tmp_path / "origin"
    (origin / "veris").mkdir(parents=True)
    for name in ("setup.cfg", "setup.py", "versioneer.py"):
        shutil.copyfile(root / name, origin / name)

    def git(*arguments: str, cwd: Path = origin) -> str:
        """Keep fixture Git operations local and independent of user config."""
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Documentation test",
                "-c",
                "user.email=docs@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                *arguments,
            ],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--initial-branch=main")
    # Source distributions freeze _version.py; regenerate the Git-aware module.
    subprocess.run(
        [sys.executable, "versioneer.py", "setup"],
        cwd=origin,
        check=True,
        capture_output=True,
        text=True,
    )
    git("add", ".")
    git("commit", "-m", "Release fixture")
    git("tag", "-a", "v0.2.0", "-m", "Release 0.2.0")
    for _ in range(distance):
        git("commit", "--allow-empty", "-m", "Development fixture")

    checkout = tmp_path / "checkout"
    git(
        "clone",
        "--no-tags",
        *(["--depth=1"] if shallow else []),
        origin.as_uri(),
        str(checkout),
    )
    assert not git("tag", "--list", cwd=checkout)
    with (root / ".readthedocs.yml").open(encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    for command in configuration["build"].get("jobs", {}).get("post_checkout", []):
        subprocess.run(
            ["sh", "-eu", "-c", command],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        )

    version = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                "print(runpy.run_path('veris/_version.py')['get_versions']()['version'])"
            ),
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = (
        "0.2.0"
        if distance == 0
        else f"0.2.0+2.g{git('rev-parse', '--short=7', 'HEAD', cwd=checkout)}"
    )
    assert version == expected, f"ERROR documentation release: {version} != {expected}"
