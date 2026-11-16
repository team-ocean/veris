"""Render verified Veris setup output as publication and web figures.

Run the documented CLI simulations first, then pass their output directory.
Snapshots contain physical (x, y) cells; maps transpose once and mask land.
Growth uses grid-cell mean hIceMean through the first exact zero, inclusive.
Matplotlib supplies outlined vector lettering, flow overlays and compact
rasterized cell maps inside standalone SVG files for web/publication delivery.
This script does not initialize a model or change any physical parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import h5netcdf
import numpy as np
from numpy.typing import NDArray


def growth_window(
    days: NDArray[Any], thickness: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Select the first ice-free endpoint without plotting subsequent samples."""
    if days.ndim != 1 or thickness.shape != days.shape or not len(days):
        raise ValueError("growth days and thickness must be matching nonempty vectors")
    if not np.isfinite(days).all() or not np.isfinite(thickness).all():
        raise ValueError("growth samples must be finite")
    if np.any(thickness < 0) or np.any(np.diff(days) <= 0):
        raise ValueError("growth thickness must be nonnegative and time increasing")
    zero = np.flatnonzero(thickness == 0)
    if not len(zero):
        raise ValueError("growth history never reaches zero ice thickness")
    stop = int(zero[0]) + 1
    return days[:stop], thickness[:stop]


def sha256(path: Path) -> str:
    """Identify the exact raw data or delivered figure."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_output(path: Path, stream: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a physical-cell output stream and its resolved experiment metadata."""
    with h5netcdf.File(path, "r") as source:
        experiment = json.loads(source.attrs["experiment"])
        group = source.groups[stream]
        values = {name: np.asarray(field[:]) for name, field in group.variables.items()}
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite {name} in {path}")
    return experiment, values


def export_figure(figure: Any, stem: Path) -> dict[str, str]:
    """Save only portable SVG with vector lettering/overlays and a 300 dpi map."""
    target = stem.with_suffix(".svg")
    figure.savefig(target, dpi=300, facecolor="white", metadata={"Date": None})
    return {target.name: sha256(target)}


def center_ocean_velocity(
    u: NDArray[Any], v: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Average west/east U and south/north V faces onto native (x, y) centers."""
    if u.ndim != 2 or v.shape != u.shape:
        raise ValueError("ocean components must have matching two-dimensional shapes")
    return 0.5 * (u + np.roll(u, -1, axis=0)), 0.5 * (v + np.roll(v, -1, axis=1))


def draw_forcing(
    axes: Any,
    x: NDArray[Any],
    y: NDArray[Any],
    wet: NDArray[Any],
    u_ocean: NDArray[Any],
    v_ocean: NDArray[Any],
    u_wind: NDArray[Any],
    v_wind: NDArray[Any],
) -> dict[str, Any]:
    """Draw sparse real ocean streamlines and a calibrated atmospheric quiver."""
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    ocean_color, wind_color = "#009e73", "#d55e00"
    uc, vc = center_ocean_velocity(u_ocean, v_ocean)
    stationary = not np.any((np.hypot(uc, vc) > 0) & wet)
    if stationary:
        axes.text(
            0.97,
            0.97,
            "Ocean: stationary",
            transform=axes.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#006f51",
            zorder=6,
            bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "none", "pad": 2},
        )
    else:
        seed_x, seed_y = np.meshgrid(
            np.linspace(x[0], x[-1], 8)[1:-1],
            np.linspace(y[0], y[-1], 8)[1:-1],
        )
        axes.streamplot(
            x,
            y,
            np.ma.masked_where(~wet, uc).T,
            np.ma.masked_where(~wet, vc).T,
            color=to_rgba(ocean_color, 0.6),
            density=0.65,
            linewidth=0.7,
            arrowsize=0.7,
            minlength=0.15,
            maxlength=2.5,
            start_points=np.column_stack((seed_x.ravel(), seed_y.ravel())),
        )
        axes.legend(
            handles=[
                Line2D(
                    [0], [0], color=ocean_color, linewidth=0.8, label="Ocean current"
                )
            ],
            loc="upper left",
            fontsize=7,
            framealpha=0.85,
            borderpad=0.35,
        )
    nx, ny = wet.shape
    ix = np.linspace(nx // 16, nx - 1 - nx // 16, min(8, nx), dtype=int)
    iy = np.linspace(ny // 20, ny - 1 - ny // 20, min(10, ny), dtype=int)
    xx, yy = np.meshgrid(x[ix], y[iy], indexing="ij")
    selected = np.ix_(ix, iy)
    valid = wet[selected]
    quiver = axes.quiver(
        xx[valid],
        yy[valid],
        u_wind[selected][valid],
        v_wind[selected][valid],
        color=wind_color,
        alpha=0.65,
        angles="xy",
        scale_units="width",
        scale=90,
        width=0.003,
        pivot="mid",
        headwidth=3.8,
        headlength=4.5,
        zorder=4,
    )
    key = axes.quiverkey(
        quiver,
        0.9,
        0.045,
        5,
        "5 m/s wind",
        labelpos="W",
        labelcolor=wind_color,
        fontproperties={"size": 7},
    )
    key.text.set_bbox(
        {"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1}
    )
    return {
        "ocean_stationary": stationary,
        "ocean_color": ocean_color,
        "wind_color": wind_color,
        "ocean_alpha": 0.6,
        "wind_alpha": 0.65,
        "streamline_density": 0.65,
        "streamline_seed_grid": [6, 6],
        "ocean_centering": "arithmetic means of adjacent native U/V faces",
        "wind_vectors": int(valid.sum()),
        "wind_reference_m_per_s": 5,
        "ocean_speed_max_m_per_s": float(np.hypot(uc, vc)[wet].max()),
    }


def plot_concentration(data: Path, output: Path, name: str) -> dict[str, Any]:
    """Plot final concentration on the exact physical basin, masking land."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    path = data / f"{name}.nc"
    experiment, values = read_output(path, "snapshot")
    nx, ny = (experiment["grid"][axis] for axis in ("nx", "ny"))
    area, mask = values["Area"][0], values["maskInC"][0]
    if area.shape != (nx, ny) or not np.isin(mask, [0, 1]).all():
        raise ValueError("snapshot must contain physical cells and a binary land mask")
    wet = mask != 0
    if np.any(area[wet] < 0) or np.any(area[wet] > 1):
        raise ValueError("raw ice concentration lies outside zero to one")
    scenario = experiment["scenario"]
    velocity_source = path
    if not all(field in values for field in ("uOcean", "vOcean", "uWind", "vWind")):
        velocity_source = data / f"{name}-forcing.nc"
        forcing_experiment, forcing = read_output(velocity_source, "snapshot")
        # Only constant prescribed fields may be borrowed from initialization.
        if (
            forcing_experiment["grid"] != experiment["grid"]
            or forcing_experiment["scenario"] != scenario
            or forcing_experiment["forcing"] != experiment["forcing"]
            or experiment["forcing"]["series"]
        ):
            raise ValueError(
                "supplemental forcing must match the recorded constant experiment"
            )
        np.testing.assert_array_equal(forcing["maskInC"], values["maskInC"])
        values.update(
            {key: forcing[key] for key in ("uOcean", "vOcean", "uWind", "vWind")}
        )
    if name == "dynamics":
        dx, dy = (scenario["length"] / (n - 1) / 1000 for n in (nx, ny))
        size = (3.8, 4.05)
    else:
        dx = dy = scenario["islandGridSpacing"] / 1000
        size = (3.8, 5.35)
    figure, axes = plt.subplots(figsize=size, layout="constrained")
    palette = plt.get_cmap("Blues_r").with_extremes(
        bad="white" if name == "dynamics" else "#9a9a9a"
    )
    image = axes.pcolormesh(
        np.arange(nx + 1) * dx,
        np.arange(ny + 1) * dy,
        np.ma.masked_where(~wet, area).T,
        cmap=palette,
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=True,
    )
    axes.set(xlabel="x (km)", ylabel="y (km)", aspect="equal")
    axes.set_title(name.capitalize(), loc="left", fontweight="bold")
    axes.set_title(f"{experiment['steps']} steps", loc="right", fontsize=8)
    overlay = draw_forcing(
        axes,
        (np.arange(nx) + 0.5) * dx,
        (np.arange(ny) + 0.5) * dy,
        wet,
        *(values[field][0] for field in ("uOcean", "vOcean", "uWind", "vWind")),
    )
    if name == "island":
        axes.legend(
            handles=[Patch(facecolor="#9a9a9a", label="Land")],
            loc="upper left",
            fontsize=7,
            framealpha=0.85,
            borderpad=0.35,
        )
    bar = figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        pad=0.025,
        fraction=0.06,
        aspect=28,
        ticks=np.linspace(0, 1, 6),
    )
    bar.set_label("Ice concentration (fraction)")
    stem = output / f"{name}-concentration-300-steps"
    files = export_figure(figure, stem)
    plt.close(figure)
    return {
        "experiment": experiment,
        "snapshot_sha256": sha256(path),
        "model_time_seconds": float(values["time"][0]),
        "physical_shape_xy": [nx, ny],
        "grid_spacing_m_xy": [dx * 1000, dy * 1000],
        "concentration_min": float(area[wet].min()),
        "concentration_max": float(area[wet].max()),
        "land_cells": int((~wet).sum()),
        "land_area_m2": float((~wet).sum() * dx * dy * 1e6),
        "velocity_source_sha256": sha256(velocity_source),
        "velocity_source": "final snapshot"
        if velocity_source == path
        else "constant prescribed fields, zero-step initialization of recorded definition",
        "forcing_overlay": overlay,
        "files_sha256": files,
    }


def plot_growth(data: Path, output: Path) -> dict[str, Any]:
    """Plot the 200-day history only through its first zero-thickness day."""
    import matplotlib.pyplot as plt

    path = data / "growth-history.nc"
    experiment, values = read_output(path, "instantaneous")
    history = values["hIceMean"]
    np.testing.assert_array_equal(
        history, np.broadcast_to(history[:, :1, :1], history.shape)
    )
    days = values["time"] / 86400
    np.testing.assert_array_equal(days, np.arange(201))
    shown_days, ice = growth_window(days, history[:, 0, 0])
    figure, axes = plt.subplots(figsize=(4.7, 3.1), layout="constrained")
    axes.plot(shown_days, ice, color="#145a7a", linewidth=1.5)
    axes.set(
        xlabel="Time (days)",
        ylabel="Grid-cell mean ice thickness (m)",
        xlim=(0, float(shown_days[-1])),
        ylim=(0, 1.4),
    )
    axes.set_title("Thermodynamic column", loc="left", fontweight="bold")
    axes.set_xticks([0, 30, 60, 90, 120, shown_days[-1]])
    axes.grid(axis="y", color="#dedede", linewidth=0.45)
    axes.set_axisbelow(True)
    files = export_figure(figure, output / "growth-thickness-to-zero")
    plt.close(figure)
    return {
        "experiment": experiment,
        "history_sha256": sha256(path),
        "run_steps": 200,
        "recorded_samples": len(days),
        "plotted_samples": len(ice),
        "first_zero_day": float(shown_days[-1]),
        "plotted_days": shown_days.tolist(),
        "plotted_thickness_m": ice.tolist(),
        "files_sha256": files,
    }


def main() -> None:
    """Render all three completed runs and record their exact numerical provenance."""
    import matplotlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data", type=Path, help="Directory containing the documented netCDF runs"
    )
    parser.add_argument("--output", type=Path, default=Path("docs/figures/setups"))
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("docs/validation/2026-10-27-setup-figures.json"),
    )
    args = parser.parse_args()
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "veris-setup-figures",
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {
        name: plot_concentration(args.data, args.output, name)
        for name in ("dynamics", "island")
    }
    runs["growth"] = plot_growth(args.data, args.output)
    record = {
        "date": "2026-10-28",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip(),
        "source_files_sha256": {
            "veris/setups/island.py": sha256(
                Path(__file__).resolve().parents[2] / "veris/setups/island.py"
            ),
            "veris/setups/presets.py": sha256(
                Path(__file__).resolve().parents[2] / "veris/setups/presets.py"
            ),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "h5netcdf", "matplotlib")
        },
        "plot_generator_sha256": sha256(Path(__file__)),
        "exports": {
            "format": "SVG only, outlined lettering/vector overlays and 300 dpi embedded concentration map",
            "map_limits": [0, 1],
            "colormap": "Blues_r",
        },
        "runs": runs,
        "reuse_note": "Final states/history are unchanged saved runs. The saved island profile label refined is historical; this configuration is now the sole example.",
    }
    stability = args.data / "island-stability.json"
    if stability.exists():
        record["island_stability_and_sensitivity"] = json.loads(stability.read_text())
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(record, indent=2) + "\n")
    print("Rendered three SVG setup figures with prescribed ocean/wind overlays.")


if __name__ == "__main__":
    main()
