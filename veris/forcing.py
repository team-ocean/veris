"""Host definitions of prescribed ocean and atmospheric fields.

This module introduces no physical equations. Names and units refer to VARIABLES;
arrays are global physical (x, y) fields without storage halos. Time series have
a leading timestep axis and supply the forcing immediately before that step.
These immutable host inputs are separate from the dynamic JAX arrays accepted
by pure numerical execution; constructing a definition is not differentiable.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

FORCING_FIELDS = frozenset(
    {
        "uOcean",
        "vOcean",
        "theta",
        "ocSalt",
        "ssh_an",
        "surfPress",
        "uWind",
        "vWind",
        "wSpeed",
        "Qnet",
        "Qsw",
        "SWdown",
        "LWdown",
        "ATemp",
        "aqh",
        "precip",
        "snowfall",
        "evap",
        "runoff",
    }
)
"""Prescribed State fields; evolved ice and derived dynamics are excluded."""


class _ImmutableArray(np.ndarray):
    """Prevent shape/dtype reassignment as well as content mutation of inputs."""

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("experiment input arrays are immutable; copy to modify")

    def resize(self, *args: Any, **kwargs: Any) -> None:
        """Reject ndarray's in-place shape mutation as well as attribute writes."""
        raise AttributeError("experiment input arrays are immutable; copy to modify")


def _immutable_array(array: NDArray[Any]) -> NDArray[Any]:
    """Own immutable bytes and protect the array's interpretation metadata."""
    return (
        np.frombuffer(array.tobytes(), dtype=array.dtype)
        .reshape(array.shape)
        .view(_ImmutableArray)
    )


def _freeze_fields(
    values: Mapping[str, Any], allowed: frozenset[str], *, series: bool = False
) -> Mapping[str, Any]:
    """Copy finite real input fields into scalars or immutable-backed arrays."""
    if not isinstance(values, Mapping):
        raise TypeError("fields must be a mapping of names to numeric values")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or name not in allowed:
            raise ValueError(f"unsupported field {name!r}")
        try:
            array = np.asarray(value)
        except (ValueError, TypeError) as error:
            raise TypeError(f"{name} must contain real numeric values") from error
        if array.dtype.kind not in "fiu":
            raise TypeError(f"{name} must contain real numeric values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        if series:
            if array.ndim not in (1, 3):
                raise ValueError(f"{name} must have shape (time,) or (time, nx, ny)")
            if any(size == 0 for size in array.shape):
                raise ValueError(f"{name} must have a nonempty time and spatial extent")
        elif array.ndim not in (0, 2):
            raise ValueError(f"{name} must be a scalar or a 2D physical field")
        if name in ("hIceMean", "hSnowMean", "wSpeed", "ocSalt") and np.any(array < 0):
            raise ValueError(f"{name} must be nonnegative")
        if name == "Area" and np.any((array < 0) | (array > 1)):
            raise ValueError("Area must lie between zero and one")
        if name in ("TSurf", "theta", "ATemp") and np.any(array <= 0):
            raise ValueError(f"{name} must be positive in kelvin")
        # Immutable bytes prevent reversing write=False on the owning ndarray.
        result[name] = array.item() if array.ndim == 0 else _immutable_array(array)
    return MappingProxyType(result)


def _validate_shapes(
    values: Mapping[str, Any], shape: tuple[int, int], *, series: bool = False
) -> None:
    """Reject spatial arrays that do not use global physical grid dimensions."""
    for name, value in values.items():
        actual = np.shape(value)
        spatial = actual[1:] if series else actual
        if spatial and spatial != shape:
            raise ValueError(
                f"{name} has spatial shape {spatial}; expected global physical {shape}"
            )


@dataclass(frozen=True)
class Forcing:
    """Immutable constant and timestep-indexed prescribed field values.

    Constants accept scalars or (nx, ny) arrays. Series accept (time,) or
    (time, nx, ny) arrays and override constants of the same name. All series
    share a nonzero length. Fields absent from both groups keep their current
    State value, allowing legacy recursive heat fluxes. Values use VARIABLES
    units, including kelvin and positive-upward Qnet/Qsw in W m-2.
    """

    constant: Mapping[str, Any] = field(default_factory=dict)
    series: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate names, values and time lengths, then freeze owned inputs."""
        object.__setattr__(
            self, "constant", _freeze_fields(self.constant, FORCING_FIELDS)
        )
        object.__setattr__(
            self, "series", _freeze_fields(self.series, FORCING_FIELDS, series=True)
        )
        lengths = {np.shape(value)[0] for value in self.series.values()}
        if len(lengths) > 1:
            raise ValueError("forcing series must have one shared time length")

    @property
    def length(self) -> int | None:
        """Number of available steps, or None for unlimited constant forcing."""
        return next((np.shape(value)[0] for value in self.series.values()), None)

    def validate_shape(self, shape: tuple[int, int]) -> None:
        """Check every spatial input against global physical (nx, ny)."""
        _validate_shapes(self.constant, shape)
        _validate_shapes(self.series, shape, series=True)

    def at(self, step: int) -> dict[str, Any]:
        """Select an absolute zero-based timestep without advancing a cursor.

        The returned mapping is a fresh host dictionary; its arrays remain
        immutable. This method is for host preparation, outside JAX transforms.
        """
        if type(step) is not int:
            raise TypeError("forcing step must be an integer")
        if step < 0:
            raise ValueError("forcing step must be nonnegative")
        if self.length is not None and step >= self.length:
            raise ValueError(f"forcing exhausted at step {step}; length {self.length}")
        return {
            **self.constant,
            **{name: value[step] for name, value in self.series.items()},
        }
