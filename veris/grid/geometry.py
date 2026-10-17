"""Cold host ocean geometry schema, originally veris.model.types.OceanGeometry.

Masks include a vertical axis; horizontal arrays and spacing vectors include
halos. The ocean adapter validates metrics before allocating or taking inverses.
This schema selects no devices and adds no leaves to the ice State PyTree.
"""

from dataclasses import dataclass

from veris.model.arrays import ArrayInput


@dataclass(frozen=True)
class OceanGeometry:
    """Read-only ocean grid inputs, independent of any ocean-model container.

    ``maskT``, ``maskU`` and ``maskV`` are volume masks; ``dxt`` and ``dxu``
    are x-spacing vectors, and ``dyt`` and ``dyu`` are y-spacing vectors (m).
    ``ht`` is depth (m), ``coriolis_t`` is Coriolis frequency (s-1), and the
    three horizontal cell-area arrays are in m2. Shapes and finite positive
    metrics are checked by the host adapter before any reciprocal is computed.
    Optional ``latitude``/``latitude_edges`` contain ny/ny+1 physical degrees,
    excluding halos; ``radius`` is a positive scalar in metres. Supply all three
    for spherical geometry. Zonal spacing vectors then specify equatorial
    distances, while meridional spacing remains physical distance.
    """

    maskT: ArrayInput
    maskU: ArrayInput
    maskV: ArrayInput
    ht: ArrayInput
    coriolis_t: ArrayInput
    dxt: ArrayInput
    dxu: ArrayInput
    dyt: ArrayInput
    dyu: ArrayInput
    area_t: ArrayInput
    area_u: ArrayInput
    area_v: ArrayInput
    latitude: ArrayInput | None = None
    latitude_edges: ArrayInput | None = None
    radius: ArrayInput | float | None = None
