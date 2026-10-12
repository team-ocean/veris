"""Previous version-1 restart metadata retains its original equation choice."""

import json
from dataclasses import replace
from pathlib import Path

import h5netcdf
import pytest

import veris
from tests.io.test_restart import _assert_same_state, _experiment


@pytest.mark.parametrize("preset", ["growth", "dynamics", "island"])
def test_restart_before_notebook_equation_selector_preserves_corrected_trajectory(
    tmp_path: Path, preset: str
) -> None:
    """Version-1 checkpoints written before the new flag always used False."""
    definition = _experiment(preset, steps=3)
    definition = replace(
        definition,
        settings={**definition.settings, "useLegacyGrowthAreaWeights": False},
    )
    path = tmp_path / "previous-v1.nc"
    with veris.Simulation.initialize(definition) as original:
        original.step()
        saved = original.state
        original.save_restart(path)
        original.run()
        expected = original.state
    # Reconstruct precisely the previous complete v1 metadata inventory.
    with h5netcdf.File(path, "r+") as file:
        manifest = json.loads(file.attrs["experiment"])
        configuration = json.loads(file.attrs["configuration"])
        assert manifest["settings"].pop("useLegacyGrowthAreaWeights") is False
        assert configuration.pop("useLegacyGrowthAreaWeights") is False
        file.attrs["experiment"] = json.dumps(manifest)
        file.attrs["configuration"] = json.dumps(configuration)
    with veris.Simulation.from_restart(path) as resumed:
        assert not resumed.configuration.useLegacyGrowthAreaWeights
        _assert_same_state(resumed.state, saved)
        resumed.run()
        _assert_same_state(resumed.state, expected)


@pytest.mark.parametrize(
    "omission", ["settings_only", "configuration_only", "other_key"]
)
def test_restart_selector_migration_rejects_incomplete_other_metadata(
    tmp_path: Path, omission: str
) -> None:
    """The single known version-1 migration must not enable arbitrary defaults."""
    restart_path = tmp_path / "incomplete.nc"
    with veris.Simulation.initialize(_experiment("growth", steps=1)) as original:
        original.save_restart(restart_path)
    with h5netcdf.File(restart_path, "r+") as file:
        manifest = json.loads(file.attrs["experiment"])
        configuration = json.loads(file.attrs["configuration"])
        if omission != "configuration_only":
            manifest["settings"].pop("useLegacyGrowthAreaWeights")
        if omission != "settings_only":
            configuration.pop("useLegacyGrowthAreaWeights")
        if omission == "other_key":
            manifest["settings"].pop("nITC")
            configuration.pop("nITC")
        file.attrs["experiment"] = json.dumps(manifest)
        file.attrs["configuration"] = json.dumps(configuration)
    with pytest.raises(ValueError, match="incomplete|incompatible"):
        veris.Simulation.from_restart(restart_path)
