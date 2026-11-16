"""Explicit compact island controls for numerical tests and archived oracles.

These inputs retain the former 8 km grid, two-cell island and fixed five-step
EVP solver. Public island defaults instead describe the resolved 800 m example.
This helper keeps small tests and original trajectory archives independent of
changes to that public launch configuration; it is not a public setup profile.
"""

from typing import Any

from veris import Experiment, Grid


def compact_island(**controls: Any) -> Experiment:
    """Build the original compact numerical case with explicit override layers."""
    controls.setdefault("grid", Grid(8, 12))
    controls["scenario"] = {
        "islandGridSpacing": 8000.0,
        "islandHalfWidthCells": 1,
        "islandTimeStep": 600.0,
        "islandEVPsteps": 5,
        **controls.get("scenario", {}),
    }
    controls["settings"] = {
        "useAdaptiveEVP": False,
        **controls.get("settings", {}),
    }
    return Experiment.builtin("island", **controls)


def compact_builtin(preset: str, **controls: Any) -> Experiment:
    """Keep variable-preset test helpers compact in their island branch."""
    if preset == "island":
        return compact_island(**controls)
    return Experiment.builtin(preset, **controls)
