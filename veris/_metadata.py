"""Immutable scalar metadata and host validation shared by configuration registries."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import Field
from numbers import Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from veris.configuration import Setting
    from veris.physical_constants import PhysicalConstant

# An explicit default keeps dataclass constructor arguments optional to static
# type checkers. The decorator replaces this marker before dataclass runs.
FROM_REGISTRY: Any = object()


def precision_scalar(value: Any, dtype: str, name: str) -> Any:
    """Round a host coefficient once, rejecting overflow and nonzero underflow."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.dtype(dtype).type(value)
    if not np.isfinite(result) or (value != 0 and result == 0):
        raise ValueError(f"{name} is not representable in dtype {dtype}")
    return result


def registry_defaults[T](
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
    """Reject invalid static inputs and normalize real coefficients to the model precision.

    Booleans must be actual bools and counts actual ints. Real-valued coefficients
    accept finite host real scalars, including integers; arrays and tracers are
    not static configuration. Positive names protect denominators and cutoffs.
    """
    from veris.configuration import SETTINGS

    dtype = cast(str, getattr(instance, "dtype", SETTINGS["dtype"].default))
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
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
                instance,
                name,
                tuple(precision_scalar(element, dtype, name) for element in value),
            )
            continue
        if metadata.type is float:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(instance, name, precision_scalar(value, dtype, name))
        elif type(value) is not metadata.type:
            raise TypeError(f"{name} must be {metadata.type.__name__}")
        if name in positive and value <= 0:
            raise ValueError(f"{name} must be positive")


def validate_derived(instance: object, names: tuple[str, ...]) -> None:
    """Reject overflow in exact dependencies computed from finite input scalars."""
    from veris.configuration import SETTINGS

    for name in names:
        object.__setattr__(
            instance,
            name,
            precision_scalar(
                getattr(instance, name),
                cast(str, getattr(instance, "dtype", SETTINGS["dtype"].default)),
                name,
            ),
        )
        if not math.isfinite(getattr(instance, name)):
            raise ValueError(f"{name} must be finite")
