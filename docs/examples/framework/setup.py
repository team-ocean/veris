"""Build a prescribed-ocean coupled experiment without ocean-model dependencies."""

import numpy as np

from veris import Experiment, Forcing, Grid, OceanGeometry


def build_experiment() -> Experiment:
    """Return host geometry/inputs; the framework owns resources and integration."""
    grid = Grid(8, 12)
    wet = np.ones((grid.nx, grid.ny))
    wet[3:5, 5:7] = 0.0
    wet[2, 4] = 0.5
    spacing = 8000.0
    horizontal = (grid.nx, grid.ny)
    geometry = OceanGeometry(
        maskT=wet[..., None],
        maskU=(wet * np.roll(wet, 1, axis=0))[..., None],
        maskV=(wet * np.roll(wet, 1, axis=1))[..., None],
        ht=np.full(horizontal, -100.0),
        coriolis_t=np.full(horizontal, 1e-4),
        dxt=np.full(grid.nx, spacing),
        dxu=np.full(grid.nx, spacing),
        dyt=np.full(grid.ny, spacing),
        dyu=np.full(grid.ny, spacing),
        area_t=np.full(horizontal, spacing**2),
        area_u=np.full(horizontal, spacing**2),
        area_v=np.full(horizontal, spacing**2),
    )
    return Experiment(
        grid=grid,
        geometry=geometry,
        physics="coupled",
        steps=3,
        settings={"deltatDyn": 600.0, "deltatTherm": 600.0, "nEVPsteps": 2},
        initial_conditions={
            "hIceMean": 1.0,
            "hSnowMean": 0.05,
            "Area": 0.8,
            "TSurf": 260.0,
        },
        forcing=Forcing(
            constant={
                "theta": 271.35,
                "ocSalt": 34.7,
                "uWind": 5.0,
                "vWind": 0.0,
                "ATemp": 260.0,
                "LWdown": 259.0,
                "aqh": 0.001,
                "Qnet": 100.0,
                "Qsw": 0.0,
            }
        ),
    )
