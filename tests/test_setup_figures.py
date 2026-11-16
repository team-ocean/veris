"""Scientific figure data selection must retain the first ice-free endpoint."""

import runpy
from pathlib import Path

import numpy as np
import pytest


def test_ocean_velocity_uses_adjacent_native_faces_without_swapping_axes() -> None:
    """A wrong roll direction or x/y transpose moves the displayed circulation."""
    plot = runpy.run_path(
        str(Path(__file__).parents[1] / "docs/examples/plot_setup_results.py")
    )
    u = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]])
    v = np.array([[0, 10, 20]] * 4)
    uc, vc = plot["center_ocean_velocity"](u, v)
    np.testing.assert_array_equal(uc, [[1, 1, 1], [3, 3, 3], [5, 5, 5], [3, 3, 3]])
    np.testing.assert_array_equal(vc, [[5, 15, 10]] * 4)


def test_stationary_ocean_overlay_draws_only_real_wind_vectors() -> None:
    """Zero prescribed current must not be replaced by invented flow or ice drift."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.quiver import Quiver

    plot = runpy.run_path(
        str(Path(__file__).parents[1] / "docs/examples/plot_setup_results.py")
    )
    figure, axes = plt.subplots()
    zero = np.zeros((8, 12))
    try:
        result = plot["draw_forcing"](
            axes,
            np.arange(8),
            np.arange(12),
            np.ones_like(zero, dtype=bool),
            zero,
            zero,
            np.full_like(zero, 5),
            zero,
        )
        assert result["ocean_stationary"]
        assert len(axes.collections) == 1
        assert isinstance(axes.collections[0], Quiver)
        assert np.all(axes.collections[0].U == 5)
        assert np.all(axes.collections[0].V == 0)
        assert np.asarray(axes.collections[0].U).size < zero.size
    finally:
        plt.close(figure)


def test_growth_window_stops_at_first_zero_and_keeps_that_sample() -> None:
    """Plotting beyond first disappearance or omitting its endpoint is incorrect."""
    plot = runpy.run_path(
        str(Path(__file__).parents[1] / "docs/examples/plot_setup_results.py")
    )
    days, ice = plot["growth_window"](
        np.array([0, 1, 2, 3, 4]), np.array([1.3, 0.7, 0.0, 0.0, 0.2])
    )
    np.testing.assert_array_equal(days, [0, 1, 2])
    np.testing.assert_array_equal(ice, [1.3, 0.7, 0])


def test_growth_window_rejects_a_run_without_disappearance() -> None:
    """Do not fabricate a zero endpoint when the simulation retains ice."""
    plot = runpy.run_path(
        str(Path(__file__).parents[1] / "docs/examples/plot_setup_results.py")
    )
    with pytest.raises(ValueError, match="zero"):
        plot["growth_window"](np.array([0, 1, 2]), np.array([1.3, 0.7, 0.1]))
