"""Discover and copy host-side setup templates without importing their code.

Builtin templates define the maintained presets through Experiment. Additional
directories in VERIS_SETUP_DIR contain named folders with a setup.py factory
and optional relative assets. Discovery allocates no numerical arrays. Names
must be unique across builtin and external roots; copying never merges folders.
This provides the show/copy operations described by Vercor's cli.html.
"""

import os
import shutil
from pathlib import Path


def discover_setups() -> dict[str, Path]:
    """Return sorted template names and directories, rejecting ambiguous names."""
    roots = [Path(__file__).parent / "templates"]
    roots.extend(
        Path(value).expanduser()
        for value in os.environ.get("VERIS_SETUP_DIR", "").split(os.pathsep)
        if value
    )
    setups: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"setup directory does not exist: {root}")
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not (directory / "setup.py").is_file():
                continue
            if directory.name in setups:
                raise ValueError(f"duplicate setup name {directory.name!r}: {root}")
            setups[directory.name] = directory
    return dict(sorted(setups.items()))


def copy_setup(name: str, destination: Path) -> Path:
    """Copy an entire named template to a new destination, retaining its assets."""
    setups = discover_setups()
    if name not in setups:
        raise ValueError(f"unknown setup {name!r}; available: {', '.join(setups)}")
    source = setups[name]
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("copy destination must be outside the source template")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    return destination
