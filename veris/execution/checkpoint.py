"""Checkpoint pure transitions while retaining distributed halo intermediates.

Transitions accept and return arbitrary array PyTrees; their equations and AD
rules remain unchanged. JAX rematerialization otherwise replays primal halo
permutations inside nested EVP scans during reverse AD. CPU Gloo communicator
scheduling can overlap these with independent adjoint halo exchanges. Saving
known permutation outputs removes that replay while retaining actual adjoint
communication and rematerialization of arithmetic. Residual storage grows with
both outer timestep and inner static solver iteration counts.

Single-process execution uses JAX's default checkpoint policy. This wrapper
implements the rollout storage policy, not a numerical primitive or host wait.
See https://docs.jax.dev/en/latest/gradient-checkpointing.html.
"""

from collections.abc import Callable
from typing import ParamSpec, TypeVar

import jax
from jax.extend.core import Primitive

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


def checkpoint_transition(
    transition: Callable[_Parameters, _Result],
) -> Callable[_Parameters, _Result]:
    """Checkpoint a pure transition with its original input/output PyTrees."""
    policy = _save_permutations if jax.process_count() > 1 else None
    return jax.checkpoint(transition, policy=policy)


def _save_permutations(
    primitive: Primitive, *_arguments: object, **_parameters: object
) -> bool:
    """Retain known halo exchange results across nested static solver scans."""
    return primitive.name == "ppermute"
