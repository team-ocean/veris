"""Execute the literal paired setup blueprint against a clean installed wheel."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from installed_lifecycle_probe import _block


def main() -> None:
    """Prove installed Python/CLI/config-export equivalence for all State fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    args = parser.parse_args()
    package_root, docs_root = args.package_root.resolve(), args.docs_root.resolve()
    import veris
    from veris.io import read_record

    assert Path(veris.__file__).resolve().is_relative_to(package_root), (
        "ERROR blueprint imported source instead of installed package"
    )
    documentation = docs_root / "quickstart/setup-blueprint.rst"
    environment = dict(
        os.environ,
        PYTHONPATH=str(package_root),
        PATH=str(package_root / "bin") + os.pathsep + os.environ["PATH"],
    )
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="veris-installed-framework-") as location:
        root = Path(location)
        shutil.copyfile(docs_root / "examples/framework/setup.py", root / "setup.py")
        sys.path.insert(0, str(root))
        try:
            os.chdir(root)
            namespace: dict[str, Any] = {"__name__": "__main__"}
            code = _block(documentation, "from pathlib import Path", "Run from Python")
            exec(compile(code, str(documentation), "exec"), namespace)  # noqa: S102 -- trusted repository documentation
            commands = (
                (
                    "Run from the CLI",
                    "veris run setup.py --steps 3 --backend cpu --output cli.nc",
                ),
                ("Inspect and export", "veris inspect setup.py --export resolved.json"),
            )
            for section, first in commands:
                for command in _block(documentation, first, section).splitlines():
                    if not command.strip():
                        continue
                    completed = subprocess.run(
                        shlex.split(command),
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        check=False,
                    )
                    assert completed.returncode == 0, (
                        f"ERROR installed blueprint {command}: {completed.stderr[-1800:]}"
                    )
            expected, actual = read_record("python.nc"), read_record("cli.nc")
            assert expected.time == actual.time == 1800.0
            assert len(expected.fields) == 70 and set(expected.fields) == set(
                actual.fields
            )
            import numpy as np

            errors = {
                name: float(
                    np.max(np.abs(actual.fields[name] - value) / (1 + np.abs(value)))
                )
                for name, value in expected.fields.items()
            }
            assert max(errors.values()) <= 1e-12, (
                f"ERROR installed blueprint max_scaled={max(errors.values()):.6g}"
            )
            resolved, authored = (
                veris.Experiment.from_file(path)
                for path in ("resolved.json", "authored.json")
            )
            original = namespace["build_experiment"]()
            assert resolved.to_dict() == authored.to_dict() == original.to_dict()
            assert json.loads(Path("authored.json").read_text())["settings"] == dict(
                original.settings
            )
        finally:
            os.chdir(previous)
            sys.path.pop(0)
    print(
        "installed framework blueprint: Python/CLI70 fields and authored/resolved exports match"
    )


if __name__ == "__main__":
    main()
