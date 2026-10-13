"""Grouped CLI controls reject ambiguous paths and share mapping precedence."""

from importlib import import_module

import pytest


def test_valid_grouped_paths_parse_json_values_and_replace_whole_arrays() -> None:
    parser = import_module("veris.cli.overrides")
    assert parser.parse_overrides(
        ("steps=0", "settings.enable_cyclic_y=false", "forcing.series.Qnet=[1,2]")
    ) == {
        "steps": 0,
        "settings": {"enable_cyclic_y": False},
        "forcing": {"series": {"Qnet": [1, 2]}},
    }


@pytest.mark.parametrize(
    "value",
    [
        "settings..dtype=1",
        "output.streams.0.name=1",
        "unknown.value=1",
        "settings.dtype=NaN",
        "settings.dtype",
        "settings.dtype=bad",
    ],
)
def test_invalid_paths_and_json_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        import_module("veris.cli.overrides").parse_overrides((value,))


def test_duplicate_and_parent_child_overrides_are_rejected() -> None:
    parser = import_module("veris.cli.overrides")
    for values in (("steps=0", "steps=1"), ("output=null", "output.enabled=false")):
        with pytest.raises(ValueError, match="duplicate|conflict"):
            parser.parse_overrides(values)


def test_dedicated_and_generic_controls_merge_or_reject_conflicting_aliases() -> None:
    parser = import_module("veris.cli.overrides")
    assert parser.combine_overrides(
        {"settings": {"dtype": "float64"}}, {"steps": 0}
    ) == {"settings": {"dtype": "float64"}, "steps": 0}
    with pytest.raises(ValueError, match="conflict"):
        parser.combine_overrides(
            {"settings": {"dtype": "float64"}}, {"settings": {"dtype": "float32"}}
        )
