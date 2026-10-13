"""Settings parsing stays lazy and validates the maintained CLI contract."""

from pathlib import Path

import pytest

from veris.io import OutputManager
from veris.io.cli import driver_definition, parse_options
from veris.setups.run_parallel import cli


def test_disabled_history_retains_calendar_and_origin(tmp_path: Path) -> None:
    args = parse_options(
        cli,
        [
            "--output",
            str(tmp_path / "final.nc"),
            "--calendar",
            "360_day",
            "--start-date",
            "2000-02-30",
        ],
    )
    settings = driver_definition("dynamics", "parallel", args).output
    assert settings is not None
    assert not settings.enabled and settings.calendar == "360_day"
    assert (settings.start.year, settings.start.month, settings.start.day) == (
        2000,
        2,
        30,
    )
    with OutputManager(args.netcdf or ".", settings) as manager:
        assert manager.settings == settings
    assert not (tmp_path / "final.nc").exists()


@pytest.mark.parametrize("interval", [0, -1, 601])
def test_settings_parser_rejects_unreachable_sampling(interval: int) -> None:
    args = parse_options(cli, ["--sample-seconds", str(interval)])
    with pytest.raises(ValueError, match="sampling interval.*positive"):
        driver_definition("dynamics", "parallel", args)


def test_settings_parser_rejects_history_final_collision(tmp_path: Path) -> None:
    destination = str(tmp_path / "same.nc")
    args = parse_options(cli, ["--netcdf", destination, "--output", destination])
    with pytest.raises(ValueError, match="distinct paths"):
        driver_definition("dynamics", "parallel", args)
