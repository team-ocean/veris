"""Grouped JSON CLI overrides using the authoritative host mapping merge.

Paths address experiment groups or top-level controls. Arrays replace whole
values; indexing streams/arrays is unsupported. Duplicate and parent/child
paths, including dedicated flag aliases, fail instead of relying on ordering.
"""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from veris.model.experiment_file import _nonfinite_constant, _unique_object
from veris.model.experiment_resolution import merge_overrides

_ROOTS = frozenset(
    {
        "grid",
        "settings",
        "constants",
        "physics",
        "preset",
        "scenario",
        "profile",
        "geometry",
        "initial_conditions",
        "forcing",
        "execution",
        "output",
        "steps",
    }
)


def parse_overrides(values: Sequence[str]) -> dict[str, Any]:
    """Parse PATH=JSON controls, rejecting ambiguous or malformed paths/values."""
    result: dict[str, Any] = {}
    paths: list[tuple[str, ...]] = []
    for value in values:
        path, separator, encoded = value.partition("=")
        parts = tuple(path.split("."))
        if (
            not separator
            or parts[0] not in _ROOTS
            or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) for part in parts)
        ):
            raise ValueError(
                f"invalid override path {path!r}; expected SECTION.KEY=JSON"
            )
        if any(parts[: len(old)] == old or old[: len(parts)] == parts for old in paths):
            raise ValueError(f"duplicate or conflicting override path {path!r}")
        try:
            decoded = json.loads(
                encoded,
                parse_constant=_nonfinite_constant,
                object_pairs_hook=_unique_object,
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"invalid JSON for override {path!r}: {error}") from error
        nested = decoded
        for part in reversed(parts):
            nested = {part: nested}
        result = merge_overrides(result, nested)
        paths.append(parts)
    return result


def _paths(
    values: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    """List leaf control paths while treating arrays and scalar groups as whole values."""
    result = []
    for name, value in values.items():
        path = (*prefix, name)
        result.extend(_paths(value, path) if isinstance(value, Mapping) else [path])
    return result


def combine_overrides(
    generic: Mapping[str, Any], dedicated: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge explicit CLI sources, rejecting overlapping aliases first."""
    for left in _paths(generic):
        for right in _paths(dedicated):
            if left[: len(right)] == right or right[: len(left)] == left:
                raise ValueError(
                    f"conflicting dedicated and --set override: {'.'.join(left)}"
                )
    return merge_overrides(generic, dedicated)
