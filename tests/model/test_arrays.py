"""Shared JAX/NumPy array input contracts have one cold schema owner."""

from importlib import import_module


def test_array_contract_reexports_preserve_identity() -> None:
    """Geometry and all model kernels use the existing input aliases."""
    arrays = import_module("veris.model.arrays")
    previous = import_module("veris.model.types")
    assert arrays.ArrayInput is previous.ArrayInput
    assert arrays.MaskInput is previous.MaskInput
