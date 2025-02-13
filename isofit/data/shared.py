"""
Contains shared objects for the downloader modules to use to keep the CLI consistent
"""

from functools import partial

import click

from isofit.data import download, validate

download = download.cli
validate = validate.cli

output = partial(click.option, "-o", "--output")

tag = click.option(
    "-t", "--tag", default=f"latest", help="Release tag to pull", show_default=True
)

overwrite = click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite any existing installation",
    show_default=True,
)

path = partial(click.option, "-p", "--path")

check = click.option(
    "-c",
    "--check",
    is_flag=True,
    default=False,
    help="Only check for updates",
    show_default=True,
)
