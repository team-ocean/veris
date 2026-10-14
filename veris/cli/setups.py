"""Discover, resolve and copy setup templates without importing their code.

Builtin templates define the maintained presets through Experiment. Additional
directories in VERIS_SETUP_DIR contain named folders with a setup.py factory
and optional relative assets. Discovery allocates no numerical arrays. Names
must be unique across builtin and external roots; copying never merges folders.
File paths take precedence over names; all named commands share this registry.
No physics or numerical execution is implemented here.
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


def _named_setup(name: str) -> Path:
    """Look up one unambiguous template using the common discovery contract."""
    setups = discover_setups()
    if name not in setups:
        raise ValueError(f"unknown setup {name!r}; available: {', '.join(setups)}")
    return setups[name]


def resolve_setup(source: str | Path) -> Path:
    """Resolve an existing file or a registered name to its setup.py factory.

    Explicit paths and existing local files never fall back to name lookup.
    Resolution only inspects the filesystem and does not execute factories.
    """
    path = Path(source)
    if path.is_file():
        return path
    if (
        str(source) == path.name
        and path.suffix not in {".py", ".json"}
        and not path.is_dir()
        and not path.is_symlink()
    ):
        return _named_setup(path.name) / "setup.py"
    raise FileNotFoundError(f"setup file does not exist or is not a file: {source}")


def copy_setup(name: str, destination: Path) -> Path:
    """Copy an entire named template to a new destination, retaining its assets."""
    source = _named_setup(name)
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
