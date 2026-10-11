"""Host forcing definitions reject ambiguous time/space and preserve inputs."""

from typing import Any

import numpy as np
import pytest

import veris


def test_constant_and_series_forcing_use_absolute_step_and_series_precedence() -> None:
    forcing = veris.Forcing(
        constant={"Qnet": 100, "ocSalt": 34, "uWind": [[1, 2], [3, 4]]},
        series={"Qnet": [70, 110, 85]},
    )
    forcing.validate_shape((2, 2))
    assert forcing.length == 3
    assert forcing.at(0)["Qnet"] == 70
    assert forcing.at(2)["Qnet"] == 85
    assert forcing.at(2)["ocSalt"] == 34
    np.testing.assert_array_equal(forcing.at(1)["uWind"], [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="exhausted"):
        forcing.at(3)
    assert veris.Forcing(constant={"Qnet": 100}).at(1000)["Qnet"] == 100


def test_spatial_time_series_preserve_axis_order() -> None:
    values = np.arange(24).reshape(3, 2, 4)
    forcing = veris.Forcing(series={"Qnet": values, "uWind": [1, 2, 3]})
    forcing.validate_shape((2, 4))
    np.testing.assert_array_equal(
        forcing.at(1)["Qnet"], [[8, 9, 10, 11], [12, 13, 14, 15]]
    )
    with pytest.raises(ValueError, match="Qnet.*shape"):
        forcing.validate_shape((4, 2))


def test_forcing_defensively_copies_arrays_and_mappings() -> None:
    wind = np.arange(8.0).reshape(2, 4)
    cooling = [70.0, 110.0]
    source = {"uWind": wind}
    forcing = veris.Forcing(constant=source, series={"Qnet": cooling})
    wind[:] = -1
    cooling[0] = -1
    source["uWind"] = np.ones((2, 4))
    np.testing.assert_array_equal(forcing.at(0)["uWind"], np.arange(8).reshape(2, 4))
    assert forcing.at(0)["Qnet"] == 70
    with pytest.raises(TypeError):
        forcing.constant["uWind"] = 2  # ty: ignore[invalid-assignment]
    with pytest.raises(ValueError):
        np.asarray(forcing.constant["uWind"])[0, 0] = 2
    with pytest.raises(ValueError):
        np.asarray(forcing.constant["uWind"]).setflags(write=True)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"constant": {"hIceMean": 1}}, "hIceMean"),
        ({"constant": {"unknown": 1}}, "unknown"),
        ({"constant": {"Qnet": True}}, "Qnet"),
        ({"constant": {"Qnet": "100"}}, "Qnet"),
        ({"constant": {"Qnet": 1 + 2j}}, "Qnet"),
        ({"constant": {"Qnet": np.nan}}, "Qnet"),
        ({"constant": {"Qnet": [1, 2]}}, "Qnet.*scalar.*2D"),
        ({"series": {"Qnet": 1}}, "Qnet.*time"),
        ({"series": {"Qnet": []}}, "Qnet.*nonempty"),
        ({"series": {"Qnet": [[1, 2]]}}, "Qnet.*time"),
        ({"series": {"Qnet": [1, 2], "Qsw": [3]}}, "length"),
        ({"constant": {"theta": 0}}, "theta.*positive"),
        ({"constant": {"wSpeed": -1}}, "wSpeed.*nonnegative"),
    ],
)
def test_invalid_forcing_fails_with_field_context(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        veris.Forcing(**kwargs)


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_invalid_forcing_position_is_rejected(index: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="step"):
        veris.Forcing().at(index)


def test_constant_shape_is_global_physical_without_halos() -> None:
    forcing = veris.Forcing(constant={"Qnet": np.ones((6, 8))})
    with pytest.raises(ValueError, match="Qnet.*shape"):
        forcing.validate_shape((2, 4))


def test_forcing_array_metadata_cannot_reinterpret_values_or_change_time_length() -> (
    None
):
    forcing = veris.Forcing(
        constant={"uWind": np.ones((2, 4))}, series={"Qnet": [70.0, 110.0]}
    )
    with pytest.raises((ValueError, AttributeError), match="immutable"):
        forcing.constant["uWind"].shape = (4, 2)
    with pytest.raises((ValueError, AttributeError), match="immutable"):
        forcing.series["Qnet"].dtype = np.float32
    with pytest.raises((ValueError, AttributeError), match="immutable"):
        forcing.series["Qnet"].resize((1, 2), refcheck=False)
    assert forcing.length == 2
    assert forcing.at(0)["Qnet"] == 70
