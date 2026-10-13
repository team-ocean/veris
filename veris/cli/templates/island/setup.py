"""Editable coupled island with prescribed ocean fields and land masks."""

from veris import Experiment, Grid


def build_experiment() -> Experiment:
    """Define three coupled steps with the island preset's original numerics."""
    return Experiment.builtin("island", grid=Grid(8, 12), steps=3)
