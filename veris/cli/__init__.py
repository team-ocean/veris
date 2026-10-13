"""Central Click commands for discovering, copying and running Veris setups.

The installed veris entry point and python -m veris.cli share this group. Host
commands reuse Experiment/Simulation and do not implement numerical kernels.
Information and copy operations never select devices or configure JAX.
"""

import logging
from pathlib import Path

import click

from veris._version import get_versions
from veris.cli.run import run_experiment
from veris.cli.setups import copy_setup, discover_setups


@click.group(invoke_without_command=True)
@click.version_option(version=get_versions()["version"], prog_name="veris")
@click.pass_context
def cli(context: click.Context) -> None:
    """Veris command-line tools."""
    if context.invoked_subcommand is None:
        click.echo(context.get_help())


@cli.command("show-setups")
def show_setups() -> None:
    """Print available preconfigured setups."""
    try:
        click.echo("\n".join(discover_setups()))
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error


@cli.command("copy-setup")
@click.argument("setup")
@click.option(
    "--to",
    type=click.Path(file_okay=False, path_type=Path),
    help="New target directory (default: ./SETUP).",
)
def copy_setup_command(setup: str, to: Path | None) -> None:
    """Copy a setup and its assets into a new editable directory."""
    try:
        destination = copy_setup(setup, Path.cwd() / setup if to is None else to)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Copied {setup} to {destination}")


@cli.command("run")
@click.argument(
    "setup_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-v",
    "--loglevel",
    type=click.Choice(["trace", "debug", "info", "warning", "error"]),
    default="info",
    show_default=True,
)
@click.option(
    "--float-type",
    type=click.Choice(["float64", "float32"]),
    help="Override model and physical-constant precision.",
)
@click.option(
    "--backend",
    type=click.Choice(["cpu", "gpu"]),
    help="Override the setup's execution backend.",
)
@click.option(
    "--mesh",
    nargs=2,
    type=click.IntRange(min=1),
    metavar="PX PY",
    default=None,
    help="Override x/y partitions; uses all selected backend devices.",
)
@click.option(
    "--steps",
    type=click.IntRange(min=0),
    help="Override the setup's total number of timesteps.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default="output.nc",
    show_default=True,
    help="New final physical-field snapshot.",
)
@click.option(
    "--netcdf",
    type=click.Path(dir_okay=False, path_type=Path),
    help="New sampled history using the setup's output schedule.",
)
@click.option(
    "--io-variables",
    help="Comma-separated snapshot fields (default: all); also selects fields for a default history.",
)
@click.option(
    "--set", "set_values", multiple=True, help="Grouped override: SECTION.KEY=JSON."
)
def run_command(
    setup_file: Path,
    loglevel: str,
    float_type: str | None,
    backend: str | None,
    mesh: tuple[int, int] | None,
    steps: int | None,
    output: Path,
    netcdf: Path | None,
    io_variables: str | None,
    set_values: tuple[str, ...],
) -> None:
    """Run a Python setup factory or a versioned JSON experiment file."""
    logging.basicConfig(
        level=5 if loglevel == "trace" else getattr(logging, loglevel.upper())
    )
    try:
        run_experiment(
            setup_file,
            output=output,
            netcdf=netcdf,
            steps=steps,
            float_type=float_type,
            backend=backend,
            mesh=mesh,
            io_variables=io_variables,
            set_values=set_values,
        )
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        raise click.ClickException(str(error)) from error
