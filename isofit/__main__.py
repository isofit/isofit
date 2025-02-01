"""ISOFIT command line interface."""

import importlib
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


class CLI(click.MultiCommand):
    """
    Custom click class to load commands at runtime. This enables optional, external
    subcommands (such as isoplots) to be inserted into the sys path.
    """

    modules = {}

    # Name of the command: module import path
    # Module must have a function named "cli"
    commands = {
        "run": "isofit.core.isofit",
        "build": "isofit.data.build_examples",
        "download": "isofit.data.download",
        "validate": "isofit.data.validate",
        "path": "isofit.data",
        "HRRR_to_modtran": "isofit.utils.add_HRRR_profiles_to_modtran_config",
        "analytical_line": "isofit.utils.analytical_line",
        "apply_oe": "isofit.utils.apply_oe",
        "6s_to_srtmnet": "isofit.utils.convert_6s_to_srtmnet",
        "empirical_line": "isofit.utils.empirical_line",
        "ewt": "isofit.utils.ewt_from_reflectance",
        "reconstruct_subs": "isofit.utils.reconstruct",
        "sun": "isofit.utils.solar_position",
        "surface_model": "isofit.utils.surface_model",
        "plot": "isoplots",
    }

    def load_modules(self):
        import traceback

        for key, path in self.commands.items():
            try:
                self.modules[key] = importlib.import_module(path)
            except Exception as e:
                print(f"\nFailed to load: {key}")
                print(traceback.format_exc())

    def invoke(self, ctx):
        ini = ctx.params.pop("ini")
        base = ctx.params.pop("base")
        section = ctx.params.pop("section")
        paths = ctx.params.pop("path")
        keys = ctx.params.pop("keys")
        save = ctx.params.pop("save")
        preview = ctx.params.pop("preview")

        env.load(ini, section)

        if base:
            env.changeBase(base)

        for key, value in paths:
            if value:
                env.changePath(key, value)

        for key, value in keys:
            env.changeKey(key, value)

        if preview:
            print(env)
        else:
            env.save(diff_only=save)

        # If an override path is provided, insert it into the sys.paths
        if env.validate("isoplots", path=env.plots, quiet=True):
            sys.path.append(env.plots)

        self.load_modules()

        super().invoke(ctx)

    def list_commands(self, ctx):
        return self.modules

    def get_command(self, ctx, name):
        if name in self.modules:
            return self.modules[name].cli


@click.group(invoke_without_command=True, cls=CLI, add_help_option=False)
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
