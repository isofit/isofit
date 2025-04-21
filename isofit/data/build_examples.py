"""
Builds the examples from their template files for a given ISOFIT ini
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace as sns

import click

from isofit.data import env

Bash = sns(
    template="""\
#!/bin/bash

# This is a generated example script to illustrate how to execute this example via the command line

# These are important to set before executing
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

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

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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

OEScript = sns(
    template="""\
#!/bin/bash

# This is a generated example script to illustrate how to execute this example via the command line

# These are important to set before executing
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

isofit apply_oe \\
  {args}\
"""
)


class Example:
    def __init__(self, name, requires, validate={}):
        """
        Parameters
        ----------
        name : str
            Name of the directory of the example in the examples directory
        requires : list
            Required downloads to validate
        validate : dict, default={}
            Additional kew-word arguments to pass to the env.validate() function
        """
        self.name = name
        self.requires = requires
        self.validate_flags = validate

    def validate(self):
        """
        Passthrough method to validate required ISOFIT downloads
        """
        print(f"Checking the required extra files are available: {self.requires}")
        return env.validate(self.requires, **self.validate_flags)

    def setPath(self, path):
        """
        Sets the working path for the example

        Parameters
        ----------
        path : pathlib.Path
            Base path to the directory in which the example is located

        Returns
        -------
        None | False
            Returns False if the example directory is not found
        """
        self.path = Path(env.examples) / self.name

        if not self.path.exists():
            print("Error: Example directory not found")
            return False

    def build(self):
        raise NotImplementedError("Example constructor class must define this function")

    def makeConfigs(self):
        """
        Creates configs based off the template files from an example directory
        """
        print(f"Generating configs")
        templates = list((self.path / "templates").glob("*"))

        if not templates:
            print(
                "Template files not found for this example, please verify the installation"
            )
            return False

        configs = self.path / "configs"
        configs.mkdir(parents=True, exist_ok=True)
        for template in templates:
            if template.is_dir():
                output = configs / template.name
                output.mkdir(parents=True, exist_ok=True)

                for tmpl in template.glob("*.json"):
                    print(f"Creating {tmpl.parent}/{tmpl.name}")
                    updateTemplate(tmpl, output)

            elif template.suffix == ".json":
                print(f"Creating {template.name}")
                updateTemplate(template, configs)


class IsofitExample(Example):
    """
    Template for building scripts that directly call the Isofit object
    """

    def build(self):
        """
        Makes a formatted bash script

        Parameters
        ----------
        example : pathlib.Path
            Path to the example root
        path : pathlib.Path
            Path to the subset of scripts to generate scripts for
        """
        if self.makeConfigs() is False:
            return

        for path in (self.path / "configs").glob("*"):
            if path.is_dir():
                print(f"Generating scripts for: {path.name}")
                self.makeScripts(path)

    def makeScripts(self, path):
        configs = list(path.glob("*"))
        surface = next(self.path.glob("configs/*surface*.json"))

        bash, pyth = [], []
        cmds = []
        for i, config in enumerate(configs):
            fmt = {
                "i": i + 1,
                "total": len(configs),
                "name": config.name,
                "config": config,
            }
            cmds.append(fmt)
            bash.append(Bash.command.format(**fmt))
            pyth.append(Pyth.command.format(**fmt))

        # Shared arguments for both scripts
        args = {"surface_name": surface.name, "surface": surface}

        # Write bash script
        args["commands"] = "\n\n".join(bash)
        file = self.path / f"{path.name}.sh"
        tmpl = Bash.template.format(**args)
        createScript(file, tmpl)

        # Write python script
        args["commands"] = "\n\n".join(pyth)
        file = self.path / f"{path.name}.py"
        tmpl = Pyth.template.format(**args)
        createScript(file, tmpl)


class ApplyOEExample(Example):
    """
    Template for building scripts that use apply_oe
    """

    def build(self):
        self.makeApplyOE(self.path)

    def makeApplyOE(self, path):
        """
        Creates apply_oe scripts using 'args' template files
        """
        tmpl = path / "templates"
        surf = next(tmpl.glob("surface.json"))

        updateTemplate(surf, path / "configs")

        for arg in tmpl.glob("*.args.json"):
            args = updateTemplate(arg)
            args = " \\\n  ".join(args)
            name = arg.name.split(".")[0]

            file = path / f"{name}.sh"
            tmpl = OEScript.template.format(args=args)

            createScript(file, tmpl)


Examples = {
    "SantaMonica": IsofitExample(
        name="20151026_SantaMonica", requires=["data", "sixs"]
    ),
    "Pasadena": IsofitExample(name="20171108_Pasadena", requires=["data"]),
    "ThermalIR": IsofitExample(name="20190806_ThermalIR", requires=["data"]),
    "AV3Cal": IsofitExample(name="20250308_AV3Cal_wltest", requires=["data"]),
    "ImageCube-small": ApplyOEExample(
        name="image_cube/small",
        requires=["sixs", "srtmnet", "image_cube"],
        validate={"size": "small"},
    ),
    "ImageCube-medium": ApplyOEExample(
        name="image_cube/medium",
        requires=["sixs", "srtmnet", "image_cube"],
        validate={"size": "medium"},
    ),
}


def update(obj, **flags):
    """
    Recursively updates string values with .format. This operation occurs in-place.

    Parameters
    ----------
    obj : dict | list
        Object to iterate over each child value and attempt to format
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


def updateTemplate(template: str, output: str = None):
    """
    Updates a given template and writes it out to another file

    Parameters
    ----------
    template : str
        Path to template file to load and update
    output : str, default=None
        Path to write the updated template to

    Returns
    -------
    config : dict
        The updated template dictionary
    """
    with open(template, "r") as file:
        config = json.load(file)

    update(config, **env, cores=os.cpu_count())

    if output:
        output.mkdir(parents=True, exist_ok=True)
        with open(output / template.name, "w") as file:
            json.dump(config, file, indent=4)

    return config


def createScript(script: str, template: dict):
    """
    Creates an executable script file for a given template and arguments

    Parameters
    ----------
    script : str
        Path to write the script to
    template : dict
        Template being used (eg. Bash or Pyth)
    """
    print(f"Creating {script}")
    with open(script, "w") as file:
        file.write(template)

    os.chmod(script, 0o744)


def build(example, validate=True):
    """
    Builds an example directory

    Parameters
    ----------
    example : pathlib.Path
        Path to the example root
    validate : bool, default=True
        Validates the required extra downloads for each example. Disabling this will
        allow examples to build but does not guarantee they will work
    """
    print(f"Building example: {example.name}")

    if validate:
        if not example.validate():
            print(
                "One or more of the above required extra downloads is not valid, please correct and try again"
            )
            return

    if example.setPath(env.examples) is False:
        return

    print(f"Building example for this system: {example.path}")
    example.build()


@click.command(name="build")
@click.option(
    "-e",
    "--example",
    type=click.Choice(list(Examples) + ["all"]),
    default="all",
    show_default=True,
)
@click.option(
    "-nv",
    "--no-validate",
    is_flag=True,
    help="Disables validating extra installs and proceeds building examples regardless",
)
def cli(example, no_validate):
    """\
    Builds the ISOFIT examples
    """
    if env.validate(["examples"]):
        if example == "all":
            print("Building all examples")
            for i, (name, example) in enumerate(Examples.items()):
                print("=" * 16 + f" Example {i+1} of {len(Examples)} " + "=" * 16)
                build(example, validate=not no_validate)
        else:
            build(Examples[example], validate=not no_validate)
    else:
        print(
            f"ISOFIT Examples are not installed correctly, please verify before building"
        )
