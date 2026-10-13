"""Host overlay helpers for authored experiment controls.

Mappings merge by key; arrays, sequences and scalars replace whole values.
Constructor inputs stay separate from precision-normalized resolution, so a
later scenario or dtype override does not inherit an earlier rounded default.
This module performs no device selection, numerical allocation or I/O.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


def freeze_controls(value: Any) -> Any:
    """Own host control containers and arrays while retaining typed metadata."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: freeze_controls(item) for name, item in value.items()}
        )
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(freeze_controls(item) for item in value))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_controls(item) for item in value)
    if isinstance(value, np.ndarray):
        from veris.model.forcing import _immutable_array

        return _immutable_array(value)
    return value


def merge_overrides(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge grouped authored values without interpreting falsy inputs as absent."""
    if not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping")
    if any(not isinstance(name, str) for name in overrides):
        raise TypeError("override keys must be strings")
    result = dict(base)
    for name, value in overrides.items():
        prior = result.get(name)
        result[name] = (
            merge_overrides(prior, value)
            if isinstance(prior, Mapping) and isinstance(value, Mapping)
            else value
        )
    return result


def _open_controls(value: Any) -> Any:
    """Expose nested dataclass groups to mapping overlays, retaining sequences."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            part.name: _open_controls(getattr(value, part.name))
            for part in fields(value)
            if part.init
        }
    if isinstance(value, Mapping):
        return {name: _open_controls(item) for name, item in value.items()}
    return value


def constructor_inputs(instance: Any) -> dict[str, Any]:
    """Return authored dataclass inputs with grouped objects open to overlays.

    Arrays retain their immutable host ownership until the new definition copies
    them. Nested output streams keep their typed durations/calendar objects;
    the authoritative experiment codec accepts those constructor values too.
    """
    result = {}
    for item in fields(instance):
        if not item.init:
            continue
        authored = getattr(instance, "_authored_controls", {})
        if (
            item.name in ("grid", "physics", "steps", "execution", "output")
            and item.name not in authored
        ):
            continue
        value = getattr(instance, item.name)
        if item.name in authored:
            value = authored[item.name]
        result[item.name] = _open_controls(value)
    return result
