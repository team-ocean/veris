"""Run the published lifecycle examples against a clean installed package.

This standalone CI probe runs outside the source tree and uses the literal
Python examples in the documentation, including exact all-field restart checks.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any


def _block(path: Path, first_line: str, section: str | None = None) -> str:
    """Read a documentation literal block without rewriting its Python code."""
    lines = path.read_text().splitlines()
    if section is not None:
        lines = lines[lines.index(section) + 1 :]
    start = lines.index("   " + first_line)
    block = []
    for line in lines[start:]:
        if line and not line.startswith("   "):
            break
        block.append(line.removeprefix("   "))
    return "\n".join(block)


def main() -> None:
    """Validate public imports and execute coupled, observed and custom examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    docs_root = args.docs_root.resolve()
    import veris

    assert Path(veris.__file__).resolve().is_relative_to(package_root), (
        "ERROR Veris imported from source instead of clean installation"
    )
    for name in veris.__all__:
        assert hasattr(veris, name), f"ERROR public import missing: {name}"
    examples = (
        (
            docs_root / "quickstart/user-guide.rst",
            "from datetime import timedelta",
            None,
        ),
        (docs_root / "quickstart/user-guide.rst", "import jax.numpy as jnp", None),
        (
            docs_root / "quickstart/user-guide.rst",
            "from veris import Experiment, Grid, Simulation",
            None,
        ),
        (docs_root / "quickstart/custom-ocean.rst", "import numpy as np", None),
        (docs_root / "api/experiment.rst", "import jax", "Explicit meshes"),
        (
            docs_root / "reference/integration.rst",
            "import jax",
            "Time-varying forcing and AD",
        ),
    )
    namespace: dict[str, Any] = {"__name__": "__main__"}
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="veris-installed-docs-") as directory:
        try:
            os.chdir(directory)
            for path, first, section in examples:
                exec(  # noqa: S102 -- Execute trusted repository documentation verbatim.
                    compile(_block(path, first, section), str(path), "exec"), namespace
                )
            import jax
            import jax.numpy as jnp
            import numpy as np

            objective, forcing = namespace["objective"], namespace["cooling"]
            gradient = namespace["sensitivity"]
            compiled = jax.jit(objective)(forcing)
            _, tangent = jax.jvp(objective, (forcing,), (jnp.ones_like(forcing),))
            _, pullback = jax.vjp(objective, forcing)
            np.testing.assert_allclose(pullback(jnp.ones_like(compiled))[0], gradient)
            np.testing.assert_allclose(tangent, jnp.sum(gradient), rtol=1e-10)
            assert gradient.shape == (3,) and np.isfinite(gradient).all()
            assert np.any(np.asarray(gradient) != 0)
        finally:
            os.chdir(previous)
    readme = docs_root.parent / "README.md"
    literal = readme.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    with tempfile.TemporaryDirectory(prefix="veris-installed-readme-") as directory:
        try:
            os.chdir(directory)
            exec(  # noqa: S102 -- Execute trusted README's literal public example.
                compile(literal, str(readme), "exec"), {"__name__": "__main__"}
            )
        finally:
            os.chdir(previous)
    print(
        "PASS clean installed coupled restart, observations, custom ocean, sharding, pure JIT/AD and public imports"
    )


if __name__ == "__main__":
    main()
