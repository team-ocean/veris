"""Editable coupled island with prescribed ocean fields and land masks."""

from veris import Experiment


def build_experiment() -> Experiment:
    """Define 300 coupled 60-second steps on the 800 m island grid."""
    return Experiment.builtin("island", profile="example")
