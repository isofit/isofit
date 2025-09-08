import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import click

from .ini import Ini

env = Ini()

# Attach download commands
import isofit.data.cli


@click.command(name="path")
@click.argument("product", type=click.Choice(env._dirs))
@click.option("-k", "--key", help="Append the product's key")
def cli(product, key=None):
    """\
    Prints the path to a specific product
    """
    path = env[product]
    if key:
        # Check if the given key exists
        if not (sub := env[key]):
            # If it doesn't, check if this key is a subkey of the product
            # eg. product=srtmnet, key=file, actual=srtmnet.file
            subkey = f"{product}.{key}"
            if not (sub := env[subkey]):
                print(
                    f"The key {key!r} does not exist in the INI nor does the subkey {subkey!r}"
                )
                return

        path += f"/{sub}"

    print(path)


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


@dev.command(
    name="install_tab_completion",
    help="Installs ISOFIT tab completion for your current shell",
)
def installTabCompletion(command: str = "isofit"):
    """
    Generates the Click shell completion script for a user's shell, if it is supported.

    Parameters
    ----------
    command : str, default="isofit"
        Name of the CLI program
    """
    shell = os.environ.get("SHELL", "")
    home = Path.home()

    # Only shells supported by Click
    if "zsh" in shell:
        shell = "zsh"
        rc_file = home / ".zshrc"
    elif "bash" in shell:
        shell = "bash"
        rc_file = home / ".bashrc"
    elif "fish" in shell:
        shell = "fish"
        rc_file = home / ".config" / "fish" / "config.fish"
    else:
        raise RuntimeError(f"Unsupported or undetected shell: {shell}")

    # Get the source command
    source = Path(sys.executable).parent / command

    # Generate the completion script
    completion_script = subprocess.run(
        f"_{command.upper()}_COMPLETE={shell}_source {source}",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    # Save the completion script
    root = home / ".isofit"
    root.mkdir(exist_ok=True)

    completion_path = root / f"{command}_completion.{shell}"
    completion_path.write_text(completion_script)

    # Append source line to shell rc if not already present
    source_line = f"\n# {command} tab completion\nsource {completion_path}\n"
    if rc_file.exists():
        content = rc_file.read_text()
        if str(completion_path) not in content:
            rc_file.write_text(content + source_line)
    else:
        rc_file.write_text(source_line)

    print(f"{command} tab completion for shell {shell} installed in {rc_file}")
    print(
        "It is recommended to set the isofit CLI into laziest mode via: isofit -k cli_laziest 1"
    )
    print(f"You must restart your shell or run: source {rc_file}")
