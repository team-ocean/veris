"""Read independently amended ocean-budget references without changing v1 data.

The v2 archive contains explicit deltas for intentional science corrections and
absolute arrays for new diagnostics. Its generator executes archived a9 source
and independent NumPy boundary bookkeeping; it imports no corrected solver.
Every unchanged field retains its original v1 per-field oracle and tolerance.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BudgetCorrections:
    """Verified, read-only correction arrays and their explicit delta/new kind."""

    arrays: dict[str, np.ndarray]
    kinds: dict[str, str]

    def expected(self, key: str, original: np.ndarray | None) -> np.ndarray:
        """Apply one declared correction or require the unchanged original value."""
        if key not in self.kinds:
            assert original is not None, f"ERROR missing independent reference {key}"
            return original
        value = self.arrays[key]
        if self.kinds[key] == "new":
            assert original is None or key.endswith(".physics_status")
            return value
        assert self.kinds[key] == "delta" and original is not None
        return original + value


def load_budget_corrections(root: Path) -> BudgetCorrections:
    """Check archive, generator and original-source provenance before reading."""
    metadata: dict[str, Any] = json.loads((root / "ocean_budgets_v2.json").read_text())
    assert metadata["version"] == 2
    assert metadata["source_commit"] == "a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e"
    assert (
        metadata["growth_sha256"]
        == "8972b924a047dc458a00bae8fa4bbf4913310bbc60a359103557cf051f53a247"
    )
    for name, expected in metadata["base_archives"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256(
            (root / "generate_budget_corrections.py.txt").read_bytes()
        ).hexdigest()
        == metadata["generator_sha256"]
    )
    archive = root / "ocean_budgets_v2.npz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["archive_sha256"]
    )
    with np.load(archive, allow_pickle=False) as stored:
        assert set(stored.files) == metadata["corrections"].keys()
        arrays = {name: stored[name] for name in stored.files}
    for array in arrays.values():
        array.setflags(write=False)
    return BudgetCorrections(arrays, metadata["corrections"])
