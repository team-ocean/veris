"""Run the reference experiment's prescribed thermodynamic ice column.

Adapted from the run_growth experiment and initialize_growth.py on the standalone
reference's jax_halo_exchange branch. Growth implements category-averaged
ice/snow thermodynamics without dynamics or transport. A uniform 2 by 2 interior
with two halos satisfies the shared allocator and is physically equivalent to
the reference one-cell column. Qnet and Qsw outputs feed the next step;
this is a prescribed experiment, not an evolving ocean.
Importing this module neither allocates model arrays nor selects a JAX backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from types import SimpleNamespace
from typing import Any

import click
import jax
import jax.numpy as jnp
import numpy as np

from veris.execution.initialization import initialize as initialize_model
from veris.grid.halos import fill_state_overlap
from veris.io.cli import (
    output_options,
    output_settings,
    parse_options,
    prepare_output_paths,
)
from veris.model.configuration import SETTINGS, Configuration
from veris.model.diagnostics import Diagnostics
from veris.model.metadata import FROM_REGISTRY, registry_defaults, validate_scalars
from veris.model.physical_constants import PhysicalConstants
from veris.model.types import Parameter, State, jit

GROWTH_SETTINGS: dict[str, Parameter] = {
    "nx": Parameter(2, int, "Uniform column interior extent along x", "1"),
    "ny": Parameter(2, int, "Uniform column interior extent along y", "1"),
    "steps": Parameter(150, int, "Default number of thermodynamic steps", "1"),
    "hIceMean": Parameter(1.3, float, "Initial grid-cell mean ice thickness", "m"),
    "hSnowMean": Parameter(0.1, float, "Initial grid-cell mean snow thickness", "m"),
    "Area": Parameter(0.9, float, "Initial ice concentration", "1"),
    "TSurf": Parameter(273.0, float, "Initial ice surface temperature", "K"),
    "wSpeed": Parameter(2.0, float, "Prescribed wind speed", "m s-1"),
    "ocSalt": Parameter(29.0, float, "Prescribed ocean salinity", "g kg-1"),
    "oceanTemperatureC": Parameter(
        -1.66, float, "Prescribed ocean temperature", "degC"
    ),
    "Qnet": Parameter(
        173.03212617345582, float, "Initial net upward heat flux", "W m-2"
    ),
    "LWdown": Parameter(80.0, float, "Prescribed downward longwave flux", "W m-2"),
    "ATemp": Parameter(253.0, float, "Prescribed air temperature", "K"),
}


@dataclass(frozen=True)
@registry_defaults({"dtype": SETTINGS["dtype"], **GROWTH_SETTINGS})
class GrowthSettings:
    """Registry-backed reference column controls kept outside differentiable State."""

    dtype: str = field(default=FROM_REGISTRY, kw_only=True)
    nx: int = FROM_REGISTRY
    ny: int = FROM_REGISTRY
    steps: int = FROM_REGISTRY
    hIceMean: float = FROM_REGISTRY
    hSnowMean: float = FROM_REGISTRY
    Area: float = FROM_REGISTRY
    TSurf: float = FROM_REGISTRY
    wSpeed: float = FROM_REGISTRY
    ocSalt: float = FROM_REGISTRY
    oceanTemperatureC: float = FROM_REGISTRY
    Qnet: float = FROM_REGISTRY
    LWdown: float = FROM_REGISTRY
    ATemp: float = FROM_REGISTRY

    def __post_init__(self) -> None:
        """Normalize reference controls to the model precision."""
        validate_scalars(self, GROWTH_SETTINGS, positive=frozenset({"nx", "ny"}))


def initialize(
    *,
    dtype: str | None = None,
    settings_overrides: Mapping[str, Any] | None = None,
    physical_overrides: Mapping[str, Any] | None = None,
    scenario_overrides: Mapping[str, Any] | None = None,
) -> tuple[State, Configuration, PhysicalConstants]:
    """Allocate the uniform reference column with separate settings and constants.

    Shared registry defaults supply daily thermodynamic steps, zero precipitation,
    humidity and shortwave fluxes, and unit ocean masks. Model and physical
    overrides are validated by the shared initializer. Enable x64 before calling
    for float64; main does this at runtime.
    """
    overrides = dict(settings_overrides or {})
    if dtype is not None:
        overrides["dtype"] = dtype
    scenario = GrowthSettings(
        dtype=overrides.get("dtype", SETTINGS["dtype"].default),
        **dict(scenario_overrides or {}),
    )
    overrides.setdefault("nx", scenario.nx)
    overrides.setdefault("ny", scenario.ny)
    overrides.setdefault("use_sharding", False)
    if overrides["use_sharding"]:
        raise ValueError("growth column initialization supports serial execution only")
    state, conf, phys = initialize_model(
        settings_overrides=overrides, physical_overrides=physical_overrides
    )
    values = {
        name: jnp.full_like(getattr(state, name), getattr(scenario, name))
        for name in (
            "hIceMean",
            "hSnowMean",
            "Area",
            "TSurf",
            "wSpeed",
            "ocSalt",
            "Qnet",
            "LWdown",
            "ATemp",
        )
    }
    values["theta"] = jnp.full_like(
        state.theta, phys.celsius2K + scenario.oceanTemperatureC
    )
    values["SeaIceLoad"] = jnp.full_like(
        state.SeaIceLoad,
        phys.rhoIce * scenario.hIceMean + phys.rhoSnow * scenario.hSnowMean,
    )
    state = replace(state, **values)
    if not conf.enable_cyclic_y:
        state = fill_state_overlap(state, conf)
    return state, conf, phys


def step_with_diagnostics(
    state: State, conf: Configuration, phys: PhysicalConstants
) -> tuple[State, Diagnostics]:
    """Advance Growth alone, preserving recursive fluxes and returning coupling outputs.

    Ocean stresses are zero because this experiment has no momentum integration.
    All outputs retain the halo-inclusive shape of the input State.
    """
    from veris.execution.advance import advance

    return advance(state, conf, phys, physics="thermodynamics")


def step(state: State, conf: Configuration, phys: PhysicalConstants) -> State:
    """Advance one thermodynamic timestep and return the calculation State."""
    result, _ = step_with_diagnostics(state, conf, phys)
    return result


compiled_step = jit(step, static_argnames=["conf", "phys"])
"""Compiled thermodynamic step with static configuration and physical constants."""


@click.command(help=__doc__)
@click.option(
    "--steps", type=click.IntRange(min=0), default=GROWTH_SETTINGS["steps"].default
)
@click.option("--backend", type=click.Choice(["cpu", "gpu"]), default="cpu")
@output_options("growth.nc", tuple(item.name for item in fields(State)))
def cli(**kwargs: Any) -> SimpleNamespace:
    """Parse the standalone experiment options."""
    return SimpleNamespace(**kwargs)


def main(argv: list[str] | None = None) -> None:
    """Integrate the reference experiment and save physical fields as netCDF."""
    args = parse_options(cli, argv)
    jax.config.update("jax_enable_x64", True)
    from veris import Execution, Experiment, Simulation

    definition = Experiment.builtin(
        "growth", steps=args.steps, execution=Execution(backend=args.backend)
    )
    definition = replace(
        definition,
        output=output_settings(args, definition.step_duration.total_seconds()),
    )
    prepare_output_paths(args)
    with Simulation.initialize(definition, output_path=args.netcdf) as simulation:
        result = simulation.run()
        arrays = {
            item.name: np.asarray(getattr(result.state, item.name))
            for item in fields(State)
        }
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise FloatingPointError("growth simulation produced nonfinite output")
        simulation.snapshot(args.output, variables=tuple(args.io_variables.split(",")))
    print(f"Saved {args.steps} growth steps to {args.output}")


if __name__ == "__main__":
    main()
