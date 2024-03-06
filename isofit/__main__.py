"""ISOFIT command line interface."""

import click

import isofit
from isofit.core.isofit import cli_run
from isofit.utils.add_HRRR_profiles_to_modtran_config import cli_HRRR_to_modtran
from isofit.utils.analytical_line import cli_analytical_line
from isofit.utils.apply_oe import cli_apply_oe
from isofit.utils.convert_6s_to_srtmnet import cli_6s_to_srtmnet
from isofit.utils.downloads import cli_download
from isofit.utils.empirical_line import cli_empirical_line
from isofit.utils.ewt_from_reflectance import cli_ewt
from isofit.utils.solar_position import cli_sun
from isofit.utils.surface_model import cli_surface_model


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("-v", "--version", help="Print the current version", is_flag=True)
@click.option("-p", "--path", help="Print the installation path", is_flag=True)
def cli(ctx, version, path):
    """\
    This houses the subcommands of ISOFIT
    """
    if ctx.invoked_subcommand is None:
        if version:
            click.echo(isofit.__version__)

        if path:
            click.echo(__path__[0])


# Subcommands live closer to the code and algorithms they are related to.
# Import and register each manually.
cli.add_command(cli_run)
cli.add_command(cli_HRRR_to_modtran)
cli.add_command(cli_analytical_line)
cli.add_command(cli_ewt)
cli.add_command(cli_apply_oe)
cli.add_command(cli_sun)
cli.add_command(cli_download)
cli.add_command(cli_6s_to_srtmnet)
cli.add_command(cli_surface_model)
cli.add_command(cli_empirical_line)


if __name__ == "__main__":
    cli()
