"""Run the reference dynamics case on a local or distributed JAX mesh.

Adapted from ``jax_halo_exchange:run_parallel.py``. Global physical nx/ny are
partitioned by initialize_experiment; each device stores its own two-cell halos.
The timed integration compiles scan shapes before executing the trajectory.
Output contains
the nine reference fields after removing every partition's halos, in (x, y)
order. CPU ranks use one device each; local runs may expose several devices via
JAX_NUM_CPU_DEVICES. GPU execution targets the local NVIDIA CUDA devices.
Distributed initialization precedes backend allocation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import click

from veris.io.cli import (
    console_main,
    driver_definition,
    output_options,
    parameter_namespace,
    parse_options,
    run_driver,
)

OUTPUT_FIELDS = (
    "hIceMean",
    "Area",
    "hSnowMean",
    "uIce",
    "vIce",
    "uWind",
    "vWind",
    "uOcean",
    "vOcean",
)

__all__ = ["main"]


@click.command(help=__doc__)
@click.option("--nx", type=click.IntRange(min=2))
@click.option("--ny", type=click.IntRange(min=2))
@click.option("--mesh", nargs=2, type=click.IntRange(min=1), metavar="PX PY")
@click.option("--steps", type=click.IntRange(min=0))
@click.option("--evp-steps", type=click.IntRange(min=1))
@click.option("--backend", type=click.Choice(["cpu", "gpu"]))
@click.option(
    "--set", "set_values", multiple=True, help="Grouped override: SECTION.KEY=JSON."
)
@output_options("parallel.nc", OUTPUT_FIELDS)
def cli(**kwargs: Any) -> SimpleNamespace:
    """Parse explicit global grid/device controls without launching resources."""
    return parameter_namespace(kwargs)


def main(argv: list[str] | None = None) -> None:
    """Translate the parallel profile and let the public runner own resources."""
    args = parse_options(cli, argv)
    result = run_driver(driver_definition("dynamics", "parallel", args), args)
    assert result.execution_metadata is not None
    metadata = result.execution_metadata
    if metadata["process_index"] == 0:
        px, py = metadata["partitions"]
        print(
            f"mesh={px}x{py} backend={metadata['backend']} steps={result.completed_steps}"
        )
        print(
            f"compilation={result.compilation_seconds:.3f}s integration={result.integration_seconds:.3f}s output={args.output}"
        )


if __name__ == "__main__":
    console_main(main)
