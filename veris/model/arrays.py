"""Shared JAX and NumPy array inputs at numerical and grid schema boundaries.

These aliases retain the original veris.model.types accepted array/mask types.
They own no model settings, geometry or device resources.
"""

import numpy as np
from jax import Array
from numpy.typing import NDArray

type ArrayInput = Array | NDArray[np.number]
"""Indexable numerical input accepted by JAX at a compiled-kernel boundary."""


type MaskInput = ArrayInput | NDArray[np.bool_]
"""Indexable masks and halo fields may use NumPy boolean arrays as well."""
