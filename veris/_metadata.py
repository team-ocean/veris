"""Immutable scalar metadata and host validation shared by configuration registries."""

import math
from collections.abc import Callable, Mapping
from dataclasses import Field
from numbers import Real
from typing import Any, NamedTuple, TypeVar

T = TypeVar("T")

# An explicit default keeps dataclass constructor arguments optional to static
# type checkers. The decorator replaces this marker before dataclass runs.
FROM_REGISTRY: Any = object()


class Setting(NamedTuple):
    """Default, scalar type and human-readable description of a model setting."""

    default: float | int | bool
    type: type[float] | type[int] | type[bool]
    description: str
    units: str = ""


class PhysicalConstant(NamedTuple):
    """Default, scalar type and description of a physical or empirical constant."""

    default: float | tuple[float, ...]
    type: type[float] | type[tuple[float, ...]]
    description: str
    units: str = ""


def registry_defaults(
    registry: Mapping[str, Setting | PhysicalConstant],
) -> Callable[[type[T]], type[T]]:
    """Populate annotated fields before applying the standard dataclass decorator.

    Use ``FROM_REGISTRY`` for regular defaults and ``field(init=False)`` for
    derived quantities. Field order and dataclass options are preserved; defaults
    are copied at class definition time, not read from mutable metadata per call.
    """

    def decorate(cls: type[T]) -> type[T]:
        if cls.__annotations__.keys() != registry.keys():
            raise ValueError("registry keys must match annotated class fields")
        for name, metadata in registry.items():
            attribute = getattr(cls, name)
            if isinstance(attribute, Field):
                attribute.default = metadata.default
            else:
                setattr(cls, name, metadata.default)
        return cls

    return decorate


def validate_scalars(
    instance: object,
    registry: Mapping[str, Setting | PhysicalConstant],
    *,
    positive: frozenset[str],
) -> None:
    """Reject invalid static inputs and normalize real coefficients to Python floats.

    Booleans must be actual bools and counts actual ints. Real-valued coefficients
    accept finite host real scalars, including integers; arrays and tracers are
    not static configuration. Positive names protect denominators and cutoffs.
    """
    for name, metadata in registry.items():
        value = getattr(instance, name)
        if metadata.type is tuple:
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            for element in value:
                if isinstance(element, bool) or not isinstance(element, Real):
                    raise TypeError(f"{name} entries must be real scalars")
                if not math.isfinite(element):
                    raise ValueError(f"{name} entries must be finite")
            object.__setattr__(
                instance, name, tuple(float(element) for element in value)
            )
            continue
        if metadata.type is float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(instance, name, value)
        elif type(value) is not metadata.type:
            raise TypeError(f"{name} must be {metadata.type.__name__}")
        if name in positive and value <= 0:
            raise ValueError(f"{name} must be positive")


def validate_derived(instance: object, names: tuple[str, ...]) -> None:
    """Reject overflow in exact dependencies computed from finite input scalars."""
    for name in names:
        if not math.isfinite(getattr(instance, name)):
            raise ValueError(f"{name} must be finite")
