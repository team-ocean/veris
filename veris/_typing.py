"""Array inputs and signature-preserving JAX compiled callables."""

from collections.abc import Callable, Sequence
from typing import ParamSpec, Protocol, TypeVar, cast

import jax
import numpy as np
from jax import Array
from jax.stages import Lowered, Traced
from numpy.typing import NDArray

type ArrayInput = Array | NDArray[np.number]
"""Indexable numerical input accepted by JAX at a compiled-kernel boundary."""


type MaskInput = ArrayInput | NDArray[np.bool_]
"""Indexable masks and halo fields may use NumPy boolean arrays as well."""


# JAX 0.11.1's JitWrapped signature erases parameter types. This boundary
# preserves them without wrapping the compiled call or altering its execution.
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)
R = TypeVar("R")


class Kernel(Protocol[P, R_co]):
    """A compiled callable with its original signature and lowering interface."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> Lowered: ...

    def trace(self, *args: P.args, **kwargs: P.kwargs) -> Traced: ...

    def eval_shape(self, *args: P.args, **kwargs: P.kwargs) -> object:
        """Return the corresponding shape PyTree, whose container is caller-defined."""
        ...

    def clear_cache(self) -> None: ...

    @property
    def __wrapped__(self) -> Callable[P, R_co]: ...


def jit(
    function: Callable[P, R],
    *,
    static_argnames: str | Sequence[str] | None = None,
) -> Kernel[P, R]:
    """Compile with JAX while retaining the input callable's static signature."""
    return cast(Kernel[P, R], jax.jit(function, static_argnames=static_argnames))
