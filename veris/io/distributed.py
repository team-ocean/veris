"""Collect selected Veris fields, removing every partition's storage halos.

All ranks enter the collector at the same model time. JAX gathers the global
physical array collectively; only process zero returns data for h5netcdf writing.
The halo operator is shared with the maintained reference parallel driver.
"""

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh

from veris._typing import State
from veris.io.guard import require_host
from veris.io.storage import ArrayFields, Collector


def distributed_collector(mesh: Mesh) -> Collector:
    """Build the collective callback accepted by OutputManager and write_snapshot."""

    def collect(
        source: State | ArrayFields, names: tuple[str, ...], include_halos: bool
    ) -> dict[str, Any] | None:
        require_host()
        from jax.experimental import multihost_utils

        from veris.setups.run_parallel import remove_halos

        result = {}
        for name in names:
            array = (
                source[name] if isinstance(source, Mapping) else getattr(source, name)
            )
            if not include_halos:
                array = remove_halos(array, mesh)
            global_array = multihost_utils.process_allgather(array, tiled=True)
            if jax.process_index() == 0:
                result[name] = np.asarray(global_array)
        return result if jax.process_index() == 0 else None

    return collect
