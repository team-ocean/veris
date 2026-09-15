"""Keep host I/O out of JAX transformations, including constant-only calls.

JAX has no public general tracing-state query. This single compatibility boundary
uses the JAX core predicate also used internally to guard host-only operations.
Regression tests exercise grad, JVP and JIT for the supported JAX dependency.
Explicit AD-disabled output remains the primary integration contract.
"""

from jax._src import core


def under_transform() -> bool:
    """Return whether a JAX transformation is active on this thread."""
    return not core.trace_state_clean()


def require_host() -> None:
    """Reject file effects before validation, array transfer or file creation."""
    if under_transform():
        raise RuntimeError("Veris I/O must run outside JAX transformations")
