"""
Builds the examples from their template files for a given ISOFIT ini
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace as sns

import click

from isofit.data import env

Examples = {
    "SantaMonica": sns(name="20151026_SantaMonica", requires=["data", "sixs"]),
    "Pasadena": sns(name="20171108_Pasadena", requires=["data", "modtran"]),
    "ThermalIR": sns(name="20190806_ThermalIR", requires=["data", "modtran"]),
    "ImageCube": sns(name="image_cube", requires=["sixs", "srtmnet"]),
    "Hypertrace": sns(name="py-hypertrace", requires=["hypertrace", "sixs", "srtmnet"]),
}

Bash = sns(
    template="""\
#!/bin/bash

# This is a generated example script to illustrate how to execute this example via the command line

# Build a surface model first
echo 'Building surface model: {surface_name}'
isofit surface_model {surface}

# Now run retrievals
{commands}
""",
    command="""\
echo 'Running {i}/{total}: {name}'
isofit run --level DEBUG {config}\
""",
)

Pyth = sns(
    template="""\
#!/usr/bin/env python

from isofit.core.isofit import Isofit
from isofit.utils import surface_model

# Build the surface model
print('Building surface model: {surface_name}')
surface_model('{surface}')

# Now run retrievals
{commands}
""",
    command="""\
print('Running {i}/{total}: {name}')
model = Isofit('{config}')
model.run()
del model\
""",
)


def update(obj, **flags):
    """
    Recursively updates string values with .format
    """

    def iterate(obj):
        """
        Iterates over dict or list objects
        """
        if isinstance(obj, dict):
            yield from obj.items()
        elif isinstance(obj, list):
            yield from enumerate(obj)

    for key, value in iterate(obj):
        if isinstance(value, str):
            obj[key] = value.format(**flags)
        elif isinstance(value, (dict, list)):
            update(value, **flags)


def updateTemplate(template, output):
    with open(template, "r") as file:
        config = json.load(file)

    update(config, **env)

    with open(output / template.name, "w") as file:
        json.dump(config, file, indent=4)


def makeConfigs(example: Path):
    """
    Creates configs based off the template files from an example directory
    """
    templates = list((example.path / "templates").glob("*"))

    if not templates:
        print(
            "Template files not found for this example, please verify the installation"
        )
        return False

    configs = example.path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for template in templates:
        print(f"Creating {template.name}")

        if template.is_dir():
            output = configs / template.name
            output.mkdir(parents=True, exist_ok=True)

            for tmpl in template.glob("*"):
                updateTemplate(tmpl, output)
        else:
            updateTemplate(template, configs)


def createScript(script, template, args):
    print(f"Creating {script}")
    with open(script, "w") as file:
        file.write(template.template.format(**args))

    os.chmod(script, 0o744)


def makeScripts(example: Path, path: Path):
    """
    Makes a formatted bash script

    Parameters
    ----------
    example : pathlib.Path
        Path to the example root
    path : pathlib.Path
        Path to the subset of scripts to generate scripts for
    """
    configs = list(path.glob("*"))
    surface = next(example.glob("configs/*surface*.json"))

    bash, pyth = [], []
    for i, config in enumerate(configs):
        fmt = {"i": i + 1, "total": len(configs), "name": config.name, "config": config}
        bash.append(Bash.command.format(**fmt))
        pyth.append(Pyth.command.format(**fmt))

    # Shared arguments for both scripts
    args = {"surface_name": surface.name, "surface": surface}

    # Write bash script
    args["commands"] = "\n\n".join(bash)
    createScript(example / f"{path.name}.sh", Bash, args)

    # Write python script
    args["commands"] = "\n\n".join(pyth)
    createScript(example / f"{path.name}.py", Pyth, args)


def build(example):
    """
    Builds an example directory
    """
    print(f"Building example: {example.name}")

    print(f"Checking the required extra files are available: {example.requires}")
    # if not env.validate(example.requires):
    #     print('One or more of the above required extra downloads is missing, please correct and try again')
    #     return

    example.path = Path(env.examples) / example.name

    if not example.path.exists():
        print("Error: Example directory not found")
        return

    print(f"Building example for this system: {example.path}")

    print(f"Generating configs")
    if makeConfigs(example) is False:
        return

    for path in (example.path / "configs").glob("*"):
        if path.is_dir():
            print(f"Generating scripts for: {path.name}")
            makeScripts(example.path, path)


@click.command(name="build")
@click.option(
    "-e",
    "--example",
    type=click.Choice(list(Examples) + ["all"]),
    default="all",
    show_default=True,
)
def cli_build(example):
    """\
    Builds the ISOFIT examples
    """
    if env.validate(["examples"]):
        if example == "all":
            print("Building all examples")
            for name, example in Examples.items():
                build(example)
        else:
            build(Examples[example])
    else:
        print(
            f"ISOFIT Examples are not installed correctly, please verify before building"
        )
