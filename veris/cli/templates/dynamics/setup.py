"""Editable fixed-wind dynamics basin using the maintained dynamics preset."""

from veris import Experiment


def build_experiment() -> Experiment:
    """Define 100 ten-minute dynamics steps on a small rectangular CPU/GPU grid."""
    return Experiment.builtin("dynamics", profile="example")
