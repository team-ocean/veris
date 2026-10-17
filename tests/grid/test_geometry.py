"""Ocean geometry retains its schema while gaining an independent grid owner."""

from dataclasses import fields, is_dataclass
from importlib import import_module


def test_ocean_geometry_is_the_original_frozen_schema() -> None:
    """All mask, depth, Coriolis, spacing and area inputs retain their exact order."""
    cls = import_module("veris.grid.geometry").OceanGeometry
    assert is_dataclass(cls) and cls.__dataclass_params__.frozen
    assert [item.name for item in fields(cls)] == [
        "maskT",
        "maskU",
        "maskV",
        "ht",
        "coriolis_t",
        "dxt",
        "dxu",
        "dyt",
        "dyu",
        "area_t",
        "area_u",
        "area_v",
        "latitude",
        "latitude_edges",
        "radius",
    ]
    assert cls.__module__ == "veris.grid.geometry"
