"""Host preset controls adapted from the maintained standalone setups.

Scenario registries in setups remain authoritative. This module resolves only
definition defaults; existing initializers still supply the exact initial State.
No backend, arrays or physical timestep are initialized here.
"""

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from veris.model.configuration import SETTINGS
from veris.model.forcing import FORCING_FIELDS, _freeze_fields

PRESET_PHYSICS = {
    "island": "coupled",
    "dynamics": "dynamics",
    "growth": "thermodynamics",
}


def preset_controls(
    name: str, settings: Mapping[str, Any], scenario: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve numerical/scenario defaults without allocating numerical State."""
    if name not in PRESET_PHYSICS:
        raise ValueError(f"unknown builtin preset {name!r}")
    if not isinstance(scenario, Mapping):
        raise TypeError("scenario must be a mapping")
    if {"nx", "ny", "steps", "dtype"} & scenario.keys():
        raise ValueError("use Grid, Experiment steps and settings dtype for controls")
    dtype = settings.get("dtype", SETTINGS["dtype"].default)
    if name == "island":
        from veris.setups.island import (
            ISLAND_SETTINGS,
            IslandSettings,
            numerical_defaults,
        )

        registry = ISLAND_SETTINGS
        factory = IslandSettings
        arguments = {"dtype": dtype}
    elif name == "dynamics":
        from veris.setups.run_dyn import (
            DYNAMICS_NUMERICAL_DEFAULTS,
            DYNAMICS_SETTINGS,
            DynamicsSettings,
        )

        registry = DYNAMICS_SETTINGS
        factory = DynamicsSettings
        arguments = {}
    else:
        from veris.setups.run_growth import GROWTH_SETTINGS, GrowthSettings

        registry = GROWTH_SETTINGS
        factory = GrowthSettings
        arguments = {"dtype": dtype}
    unknown = scenario.keys() - registry.keys()
    if unknown:
        raise ValueError(f"unknown scenario controls: {', '.join(sorted(unknown))}")
    control = factory(**arguments, **scenario)
    resolved = {
        item.name: getattr(control, item.name)
        for item in fields(control)
        if item.name not in {"dtype", "nx", "ny", "steps"}
    }
    # GrowthSettings retains its historical validation; the public definition
    # also validates the physical quantities its scenario initializes.
    if name == "growth":
        allowed = FORCING_FIELDS | {"hIceMean", "hSnowMean", "Area", "TSurf"}
        _freeze_fields({k: v for k, v in resolved.items() if k in allowed}, allowed)
    if name == "island":
        assert isinstance(control, IslandSettings)
        defaults = numerical_defaults(control)
    elif name == "dynamics":
        defaults = dict(DYNAMICS_NUMERICAL_DEFAULTS)
    else:
        defaults = {}
    return {**defaults, **settings}, resolved
