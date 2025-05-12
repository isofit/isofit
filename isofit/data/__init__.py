import json
import logging

import click

from .ini import Ini

env = Ini()

# Attach download commands
import isofit.data.cli


@click.command(name="path")
@click.argument("product", type=click.Choice(env._dirs))
def cli(product):
    """\
    Prints the path to a specific product
    """
    print(env[product])


@click.group("dev", invoke_without_command=True, no_args_is_help=True)
def dev():
    """\
    Extra commands for developers
    """
    logging.basicConfig(level="DEBUG", format="%(levelname)-8s %(message)s")


data = click.argument("data")
save = click.option(
    "-ns",
    "--no-save",
    is_flag=True,
    default=False,
    help="Disables saving to file, will print to terminal",
)
kwargs = click.option(
    "-k",
    "--kwargs",
    multiple=True,
    nargs=2,
    help="Example: -k working_directory /path/to/output",
)


@dev.command(
    name="totmpl",
    no_args_is_help=True,
    help=env.toTemplate.__doc__,
    short_help="Convert a config to a template",
)
@data
@click.option(
    "-r", "--replace", type=click.Choice(["dirs", "keys", "None"]), default="dirs"
)
@save
@kwargs
def toTemplate(no_save, kwargs, *args, **opts):
    data = env.toTemplate(*args, save=not no_save, **dict(kwargs), **opts)

    if no_save:
        print(json.dumps(data, indent=4))


@dev.command(
    name="fromtmpl",
    no_args_is_help=True,
    help=env.fromTemplate.__doc__,
    short_help="Convert a template to a config",
)
@data
@save
@click.option("-p", "--prepend")
@kwargs
def fromTemplate(no_save, kwargs, *args, **opts):
    data = env.fromTemplate(*args, save=not no_save, **dict(kwargs), **opts)

    if no_save:
        print(json.dumps(data, indent=4))
