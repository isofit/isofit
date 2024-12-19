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
        for key, path in self.commands.items():
            if key not in self.modules:
                try:
                    self.modules[key] = importlib.import_module(path)
                except:
                    pass

    def invoke(self, ctx):
        ini = ctx.params.pop("ini")
        base = ctx.params.pop("base")
        section = ctx.params.pop("section")
        keys = ctx.params.pop("keys")
        save = ctx.params.pop("save")
        preview = ctx.params.pop("preview")

        env.load(ini, section)

        if base:
            env.changeBase(base)

        for key, value in ctx.params.items():
            if value:
                env.changePath(key, value)

        for key, value in keys:
            env.changeKey(key, value)

        if preview:
            print(env)
        else:
            env.save(diff_only=save)

        if preview:
            print(env)
        else:
            env.save(diff_only=save)

        self.load_modules()

        # If an isoplots installation was not found in the environment, attempt to retrieve it from a directory
        if "plot" not in self.modules and os.path.exists(env.plots):
            sys.path.append(env.plots)
            self.load_modules()

        super().invoke(ctx)

    def list_commands(self, ctx):
        return self.modules

    def get_command(self, ctx, name):
        if name in self.modules:
            return self.modules[name].cli


@click.group(invoke_without_command=True, cls=CLI)
@click.pass_context
@click.version_option()
@click.option("-i", "--ini", help="Override path to an isofit.ini file")
@click.option("-b", "--base", help="Override the base directory for all products")
@click.option("-s", "--section", help="Switches which section of the ini to use")
@click.option("-d", "--data", help="Override path to data directory")
@click.option("-e", "--examples", help="Override path to examples directory")
@click.option("-c", "--imagecube", help="Override path to imagecube data directory")
@click.option("-p", "--plots", help="Override path to isoplots")
@click.option("-em", "--srtmnet", help="Override path to sRTMnet installation")
@click.option("-6s", "--sixs", help="Override path to SixS installation")
# @click.option("-mt", "--modtran", help="Override path to MODTRAN installation")
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
    "-p",
    "--preview",
    is_flag=True,
    help="Prints the environment that will be used. This disables saving",
)
def cli(ctx, version, ini, base, section, keys, save, preview, **overrides):
    """\
    This houses the subcommands of ISOFIT
    """
    # Executes after CLI.invoke
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


if __name__ == "__main__":
    cli()
