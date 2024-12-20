"""ISOFIT command line interface."""

import os

# Explicitly set the number of threads to be 1, so we more effectively run in parallel
# Must be executed before importing numpy, otherwise doesn't work
# Setting in the CLI ensures this works for any script
if not os.environ.get("ISOFIT_NO_SET_THREADS"):
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import click

import isofit

# CLI imports
from isofit.core.isofit import cli_run
from isofit.data import cli_env_path, env
from isofit.data.build_examples import cli_build
from isofit.data.download import cli as dli
from isofit.utils.add_HRRR_profiles_to_modtran_config import cli_HRRR_to_modtran
from isofit.utils.analytical_line import cli_analytical_line
from isofit.utils.apply_oe import cli_apply_oe
from isofit.utils.convert_6s_to_srtmnet import cli_6s_to_srtmnet
from isofit.utils.empirical_line import cli_empirical_line
from isofit.utils.ewt_from_reflectance import cli_ewt
from isofit.utils.reconstruct import cli_reconstruct_subs
from isofit.utils.solar_position import cli_sun
from isofit.utils.surface_model import cli_surface_model


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-v", "--version", help="Print the current version", is_flag=True)
@click.option("-i", "--ini", help="Override path to an isofit.ini file")
@click.option("-b", "--base", help="Override the base directory for all products")
@click.option("-s", "--section", help="Switches which section of the ini to use")
@click.option("-d", "--data", help="Override path to data directory")
@click.option("-e", "--examples", help="Override path to examples directory")
@click.option("-c", "--imagecube", help="Override path to imagecube data directory")
@click.option("-em", "--srtmnet", help="Override path to sRTMnet installation")
@click.option("-6s", "--sixs", help="Override path to SixS installation")
# @click.option("-mt", "--modtran", help="Override path to MODTRAN installation")
@click.option(
    "--save/--no-save", " /-S", is_flag=True, default=True, help="Save the ini file"
)
def cli(ctx, version, ini, base, section, save, **overrides):
    """\
    This houses the subcommands of ISOFIT
    """
    if ctx.invoked_subcommand is None:
        if version:
            print(isofit.__version__)
            return

    env.load(ini, section)

    if base:
        env.changeBase(base)

    for key, value in overrides.items():
        if value:
            env.changePath(key, value)

    env.save(diff_only=save)


# Subcommands live closer to the code and algorithms they are related to.
# Import and register each manually.
cli.add_command(cli_run)
cli.add_command(cli_HRRR_to_modtran)
cli.add_command(cli_analytical_line)
cli.add_command(cli_ewt)
cli.add_command(cli_apply_oe)
cli.add_command(cli_sun)
cli.add_command(cli_env_path)
cli.add_command(dli.download)
cli.add_command(dli.validate)
cli.add_command(cli_build)
cli.add_command(cli_6s_to_srtmnet)
cli.add_command(cli_surface_model)
cli.add_command(cli_empirical_line)
cli.add_command(cli_reconstruct_subs)


if __name__ == "__main__":
    cli()
