"""
Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar
"""

import os
import re
import subprocess
import tarfile

import click
import requests

from isofit.data import env
from isofit.data.download import (
    cli_download,
    download_file,
    output,
    prepare_output,
    unzip,
)

URL = "https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar"


def untar(file, output):
    """ """
    with tarfile.TarFile(file) as tar:
        tar.extractall(path=output)

    os.remove(file)


def build(directory):
    """ """
    # Update the makefile with recommended flags
    file = directory / "Makefile"
    with open(file, "r") as f:
        lines = f.readlines()
        lines.insert(3, "EXTRA   = -O -ffixed-line-length-132 -std=legacy\n")

    with open(file, "w") as f:
        f.write("".join(lines))

    # Now make it
    process = subprocess.Popen(
        f"make -j {os.cpu_count()}", shell=True, stdout=subprocess.PIPE, cwd=directory
    )
    process.wait()


def download(output=None):
    """
    Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar.

    Parameters
    ----------
    output: str | None
        Path to output as. If None, defaults to the ini path.
    version: str
        Release tag to pull from the github.
    """
    click.echo(f"Downloading 6S")

    output = prepare_output(output, env.sixs)
    if not output:
        return

    file = download_file(URL, output.parent / "6S.tar")

    untar(file, output)

    click.echo("Building via make")
    build(output)

    click.echo(f"Done, now available at: {output}")


@cli_download.command(name="sixs")
@output(help="Root directory to download sixs to, ie. [path]/sixs")
def cli_examples(**kwargs):
    """\
    Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --sixs /path/sixs download sixs`: Override the ini file. This will save the provided path for future reference.
        - `isofit download sixs --output /path/sixs`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download(**kwargs)
