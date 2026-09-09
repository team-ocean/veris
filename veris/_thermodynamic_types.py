"""Array result shapes for surface heat flux and ice/snow growth kernels."""

from jax import Array

type SurfaceFluxResult = tuple[Array, Array, Array, Array, Array]
type GrowthResult = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]
