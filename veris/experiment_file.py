"""Compatibility alias for :mod:`veris.model.experiment_file`.

Import from the canonical domain module in new code. Both paths share
one module object, including private helpers and patched globals.
"""

import sys

from veris.model import experiment_file as _implementation
from veris.model.experiment_file import (
    _date,
    _duration,
    _known_keys,
    _mapping,
    _merge,
    _nonfinite_constant,
    _output,
    _output_dict,
    _plain,
    _schema_version,
    _unique_object,
    experiment_from_dict,
    experiment_to_dict,
    read_experiment,
)

__all__ = [
    "_date",
    "_duration",
    "_known_keys",
    "_mapping",
    "_merge",
    "_nonfinite_constant",
    "_output",
    "_output_dict",
    "_plain",
    "_schema_version",
    "_unique_object",
    "experiment_from_dict",
    "experiment_to_dict",
    "read_experiment",
]

sys.modules[__name__] = _implementation
