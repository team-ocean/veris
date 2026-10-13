"""Editable prescribed thermodynamic column using the maintained growth preset."""

from veris import Experiment


def build_experiment() -> Experiment:
    """Define 150 daily steps with prescribed ocean and atmospheric fields."""
    return Experiment.builtin("growth", profile="example")
