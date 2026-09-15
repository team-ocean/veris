"""Run the reference notebook's prescribed thermodynamic ice column.

Adapted from run_growth.ipynb and initialize_growth.py on the standalone
reference's jax_halo_exchange branch. Growth implements category-averaged
ice/snow thermodynamics without dynamics or transport. A uniform 2 by 2 interior
with two halos satisfies the shared allocator and is physically equivalent to
the notebook's one-cell column. Qnet and Qsw outputs feed the next step, exactly
as in the notebook; this is a prescribed experiment, not an evolving ocean.
Importing this module neither allocates model arrays nor selects a JAX backend.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from veris._metadata import FROM_REGISTRY, registry_defaults, validate_scalars
from veris._typing import Parameter, State, jit
from veris.configuration import SETTINGS, Configuration
from veris.diagnostics import Diagnostics
from veris.growth import Growth
from veris.initialization import initialize as initialize_model
from veris.io.cli import add_output_arguments, make_output, save_final
from veris.physical_constants import PhysicalConstants

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
    scenario = GrowthSettings(dtype=overrides.get("dtype", SETTINGS["dtype"].default))
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
    return replace(state, **values), conf, phys


def step_with_diagnostics(
    state: State, conf: Configuration, phys: PhysicalConstants
) -> tuple[State, Diagnostics]:
    """Advance Growth alone, preserving recursive fluxes and returning coupling outputs.

    Ocean stresses are zero because this experiment has no momentum integration.
    All outputs retain the halo-inclusive shape of the input State.
    """
    (
        ice,
        snow,
        area,
        temperature,
        freshwater,
        salt,
        shortwave,
        heat,
        load,
        penetrating,
        inverse,
    ) = Growth(state, conf, phys)
    result = replace(
        state,
        hIceMean=ice,
        hSnowMean=snow,
        Area=area,
        TSurf=temperature,
        Qsw=shortwave,
        Qnet=heat,
        SeaIceLoad=load,
        recip_hIceMean=inverse,
    )
    diagnostics = Diagnostics(
        IcePenetSW=penetrating,
        OceanStressU=jnp.zeros_like(state.uIce),
        OceanStressV=jnp.zeros_like(state.vIce),
        EmPmR=freshwater,
        forc_salt_surface=salt,
    )
    return result, diagnostics


def step(state: State, conf: Configuration, phys: PhysicalConstants) -> State:
    """Advance one thermodynamic timestep and return the calculation State."""
    result, _ = step_with_diagnostics(state, conf, phys)
    return result


compiled_step = jit(step, static_argnames=["conf", "phys"])
"""Compiled thermodynamic step with static configuration and physical constants."""


def main(argv: Sequence[str] | None = None) -> None:
    """Run the reference experiment and save days, pre-step ice and final fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=GROWTH_SETTINGS["steps"].default)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, default=Path("growth.npz"))
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    jax.config.update("jax_enable_x64", True)
    with jax.default_device(jax.devices(args.backend)[0]):
        state, conf, phys = initialize()
        history = []
        with make_output(args, conf.deltatTherm) as output:
            output.sample(state, timedelta(0))
            for iteration in range(args.steps):
                history.append(state.hIceMean[2, 2])
                state = compiled_step(state, conf, phys)
                output.sample(
                    state, timedelta(seconds=(iteration + 1) * conf.deltatTherm)
                )
        save_final(
            args,
            state,
            timedelta(seconds=args.steps * conf.deltatTherm),
            output.settings,
            conf,
            phys,
        )
        arrays = {
            item.name: np.asarray(getattr(state, item.name)) for item in fields(State)
        }
        arrays.update(
            days=np.arange(args.steps) * conf.deltatTherm / 86400.0,
            ice=np.asarray(history),
        )
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise FloatingPointError("growth simulation produced nonfinite output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as output:
            np.savez(output, **arrays)
    print(f"Saved {args.steps} growth steps to {args.output}")


if __name__ == "__main__":
    main()
