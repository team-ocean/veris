"""EVP iteration carry with concrete model state and evolving solver arrays."""

from __future__ import annotations

from jax import Array

from veris.state import State

# fori_loop promotes the initial scalar factor to a scalar JAX tracer.
type EVPCarry = tuple[
    State,
    Array,  # uIce
    Array,  # vIce
    Array,  # uIceNm1
    Array,  # vIceNm1
    Array,  # sigma1
    Array,  # sigma2
    Array,  # sigma12
    Array,  # denom1
    Array,  # denom2
    float | Array,  # EVPcFac before/during tracing
    Array,  # evpAlphaC
    Array,  # evpAlphaZ
    Array,  # evpBetaU
    Array,  # evpBetaV
    Array,  # resSig
    Array,  # resU
]
