"""Pure scan rollouts for Veris physics kernels and differentiable forcing.

The transition owns the physical equations and halo exchange; this module owns
only time iteration. State stays the scan carry, while optional diagnostics and
selected observations remain outputs. Rematerialization recomputes each step's
intermediates in reverse AD, trading computation for residual storage. It does
not remove scan's storage of carry values across time.
"""

from collections.abc import Callable
from functools import partial
from numbers import Integral
from typing import Any, TypeVar

import jax

_Carry = TypeVar("_Carry")


def step(
    initial: _Carry,
    advance: Callable[..., Any],
    steps: int,
    *,
    checkpoint: bool = True,
    inputs: Any = None,
    observe: Callable[..., Any] | None = None,
    has_aux: bool = False,
) -> Any:
    """Advance a fixed number of timesteps with ``jax.lax.scan``.

    ``advance(state)`` returns the next State. When ``inputs`` is supplied, its
    array PyTree must have leading length ``steps`` on every leaf, and the
    transition receives ``advance(state, input_slice)`` instead. Bind settings
    and constants using a closure or ``functools.partial``; dynamic forcing can
    be captured by a closure or supplied through ``inputs``.

    By default return only the final State. With a pure ``observe(state)``
    callable, return ``(final_state, observations)``; observation leaves stack
    along a new leading time axis and exclude the initial State. For transitions
    returning ``(state, diagnostics)``, pass ``has_aux=True`` and use
    ``observe(state, diagnostics)``. Unobserved diagnostics are discarded.

    ``steps``, callables and switches are static; carry and forcing arrays stay
    differentiable. Carry leaf shapes/dtypes and PyTree structure must remain
    constant. The default checkpoint wraps the scan body with ``jax.checkpoint``.
    Zero steps preserve the initial State; JAX still traces the pure transition
    and observer to infer empty observation shapes. Host I/O and mutable callbacks
    belong outside this API. For sharded physics, retain the setup's active mesh.
    """
    if isinstance(steps, bool) or not isinstance(steps, Integral):
        raise TypeError("steps must be a static nonnegative integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if not callable(advance):
        raise TypeError("advance must be a pure callable")
    if observe is not None and not callable(observe):
        raise TypeError("observe must be a pure callable or None")
    if not isinstance(checkpoint, bool) or not isinstance(has_aux, bool):
        raise TypeError("checkpoint and has_aux must be static booleans")
    if inputs is not None:
        leaves = jax.tree.leaves(inputs)
        if not leaves or any(
            not hasattr(leaf, "shape") or not leaf.shape or leaf.shape[0] != steps
            for leaf in leaves
        ):
            raise ValueError(
                "inputs must contain array leaves with leading length steps"
            )
    return _scan(initial, advance, int(steps), checkpoint, inputs, observe, has_aux)


@partial(
    jax.jit, static_argnames=("advance", "steps", "checkpoint", "observe", "has_aux")
)
def _scan(
    initial: _Carry,
    advance: Callable[..., Any],
    steps: int,
    checkpoint: bool,
    inputs: Any,
    observe: Callable[..., Any] | None,
    has_aux: bool,
) -> Any:
    """Cache compiled scans by transition and static iteration/output choices."""

    def body(state: _Carry, forcing: Any) -> tuple[_Carry, Any]:
        result = advance(state) if inputs is None else advance(state, forcing)
        if has_aux:
            state, auxiliary = result
            output = None if observe is None else observe(state, auxiliary)
        else:
            state = result
            output = None if observe is None else observe(state)
        return state, output

    transition = jax.checkpoint(body) if checkpoint else body
    final, history = jax.lax.scan(transition, initial, inputs, length=steps)
    return final if observe is None else (final, history)
