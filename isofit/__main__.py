"""ISOFIT command line interface."""

import importlib
import logging
import os
import sys

# Explicitly set the number of threads to be 1, so we more effectively run in parallel
# Must be executed before importing numpy, otherwise doesn't work
# Setting in the CLI ensures this works for any script
if not os.environ.get("ISOFIT_NO_SET_THREADS"):
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import click

import isofit
from isofit.data import env


class CLI(click.Group):
    """
    Custom click class to load commands at runtime. This enables optional, external
    subcommands (such as isoplots) to be inserted into the sys path.

    Reference: https://click.palletsprojects.com/en/stable/complex/#defining-the-lazy-group
    """

    laziest = False

    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.lazy = lazy_subcommands or {}

    def main(self, *args, **kwargs):
        """
        Loads the default ini and checks for the cli_laziest before shell tab
        completion, which may speed it up
        """
        env.load()

        self.laziest = env["cli_laziest"] == "1"

        return super().main(*args, **kwargs)

    def invoke(self, ctx):
        """
        Loads the ISOFIT ini into env and applies any overrides before invoking any
        subcommands

        Parameters
        ----------
        ctx : click.Context
            The Click context
        """
        # Pop these to not pass them along to subcommands
        ini = ctx.params.pop("ini")
        base = ctx.params.pop("base")
        section = ctx.params.pop("section")
        paths = ctx.params.pop("path")
        keys = ctx.params.pop("keys")
        save = ctx.params.pop("save")
        preview = ctx.params.pop("preview")
        self.debug = ctx.params.pop("debug")

        env.load(ini, section)

        if base:
            env.changeBase(base)

        for key, value in paths:
            if value:
                env.changePath(key, value)

        for key, value in keys:
            env.changeKey(key, value)

        # Can permanently enable the laziest flag using the ini via `isofit --keys cli_laziest 1`
        # Disable via `isofit --keys cli_laziest 0`
        self.laziest = ctx.params.pop("laziest") or env["cli_laziest"] == "1"

        # --help always disables lazier loading
        # protected_args is populated when a subcommand is invoked
        # do NOT laziest, else click.groups break
        if ctx.params["help"] or ctx.protected_args:
            self.laziest = False

        if preview:
            print(env)
        else:
            env.save(diff_only=save)

        # If an override path is provided, insert it into the sys.paths
        if env.validate("isoplots", path=env.plots, quiet=True):
            sys.path.append(env.plots)

        super().invoke(ctx)

    def list_commands(self, ctx):
        """
        List the names of available commands

        Parameters
        ----------
        ctx : click.Context
            The Click context

        Returns
        -------
        list of str
            The names of available lazy-loaded commands
        """
        base = super().list_commands(ctx)
        lazy = list(self.lazy)
        return base + lazy

    def resolve(self, cmd_name):
        """
        Resolves the import of a click function from a module string

        The import path to a Click command is to be in the format 'module:function'
        If 'function' is not provided, defaults seeking for a function named 'cli'

        Parameters
        ----------
        cmd_name : str
            Command to retrieve

        Returns
        -------
        function
            Resulting Click function imported from a module
        """
        path = self.lazy[cmd_name]
        func = "cli"
        if ":" in path:
            path, func = path.split(":")

        module = importlib.import_module(path)

        return getattr(module, func)

    def get_command(self, ctx, cmd_name):
        """
        Get a lazily-loaded command by name

        Parameters
        ----------
        ctx : click.Context
            The Click context
        cmd_name : str
            Name of the command to retrieve

        Returns
        -------
        click.Command or None
            The command object or None if not found
        """
        command = self.lazy.get(cmd_name)

        if command:
            # ctx.protected_args needs to be checked here for tab completion cases
            if self.laziest and not ctx.protected_args:
                # Just create a fake command for Click
                return click.Command(cmd_name)

            try:
                return self.resolve(cmd_name)
            except ModuleNotFoundError:
                if self.debug:
                    if cmd_name == "plot":
                        logging.exception(
                            "Isoplots does not appear to be installed, install it via `isofit download plots`"
                        )
            except:
                if self.debug:
                    logging.exception(
                        f"Failed to import {cmd_name} from {command.path}:"
                    )

        return super().get_command(ctx, cmd_name)


@click.group(
    cls=CLI,
    add_help_option=False,
    invoke_without_command=True,
    lazy_subcommands={
        "run": "isofit.core.isofit",
        "build": "isofit.data.build_examples",
        "download": "isofit.data.download",
        "validate": "isofit.data.validate",
        "path": "isofit.data",
        "dev": "isofit.data:dev",
        "HRRR_to_modtran": "isofit.utils.add_HRRR_profiles_to_modtran_config",
        "analytical_line": "isofit.utils.analytical_line",
        "apply_oe": "isofit.utils.apply_oe",
        "6s_to_srtmnet": "isofit.utils.convert_6s_to_srtmnet",
        "empirical_line": "isofit.utils.empirical_line",
        "ewt": "isofit.utils.ewt_from_reflectance",
        "reconstruct_subs": "isofit.utils.reconstruct",
        "interpolate_spectra": "isofit.utils.interpolate_spectra",
        "sun": "isofit.utils.solar_position",
        "surface_model": "isofit.utils.surface_model",
        "plot": "isoplots",
    },
)
@click.pass_context
@click.option("-i", "--ini", help="Override path to an isofit.ini file")
@click.option("-b", "--base", help="Override the base directory for all products")
@click.option("-s", "--section", help="Switches which section of the ini to use")
@click.option(
    "-p",
    "--path",
    nargs=2,
    multiple=True,
    help="Override paths with the format `-p [key] [value]`",
)
@click.option(
    "-k",
    "--keys",
    nargs=2,
    multiple=True,
    help="Override keys with the format `-k [key] [value]`",
)
@click.option(
    "--save/--no-save", " /-S", is_flag=True, default=True, help="Save the ini file"
)
@click.option(
    "--preview",
    is_flag=True,
    help="Prints the environment that will be used. This disables saving",
)
@click.option("--version", is_flag=True, help="Print the installed ISOFIT version")
@click.option("--debug", is_flag=True, help="Enables debug logging for the CLI")
@click.option(
    "--laziest",
    is_flag=True,
    help="Changes the CLI to be completely lazy, greatly speeding up the responsiveness but at the cost of not validating subcommands",
)
@click.option("--help", is_flag=True, help="Show this message and exit")
def cli(ctx, version, help, **kwargs):
    """\
    ISOFIT contains a set of routines and utilities for fitting surface, atmosphere and instrument models to imaging spectrometer data.

    \b
    Repository: https://github.com/isofit/isofit
    Documentation: https://isofit.readthedocs.io/en/latest
    Report an issue: https://github.com/isofit/isofit/issues
    """
    # invoke_without_command so that the invoke() command always gets called
    if version:
        print(isofit.__version__)

    elif help or ctx.invoked_subcommand is None:
        print(ctx.get_help())


if __name__ == "__main__":
    cli()
