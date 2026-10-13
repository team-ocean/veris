"""The published setup blueprint supplies only host geometry and physical inputs."""

import runpy
from pathlib import Path
from typing import Any

import jax
import pytest

from veris import Experiment, Grid


def test_blueprint_factory_is_cold_and_returns_the_documented_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).resolve().parents[2] / "docs/examples/framework/setup.py"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("ERROR blueprint factory selected devices")

    monkeypatch.setattr(jax, "devices", forbidden)
    definition = runpy.run_path(str(source))["build_experiment"]()
    assert isinstance(definition, Experiment) and definition.grid == Grid(8, 12)
    assert definition.physics == "coupled" and definition.steps == 3
    assert definition.geometry is not None
    assert (
        definition.configuration.deltatDyn
        == definition.configuration.deltatTherm
        == 600.0
    )
