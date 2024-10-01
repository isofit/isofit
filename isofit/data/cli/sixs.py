"""
Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar
"""

import os
import subprocess
from pathlib import Path

from isofit.data import env
from isofit.data.download import cli, download_file, prepare_output, untar

URL = "https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar"


def build(directory):
    """
    Builds a 6S directory

    Parameters
    ----------
    directory : str
        Directory with an unbuilt 6S
    """
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
    print(f"Downloading 6S")

    output = prepare_output(output, env.sixs)
    if not output:
        return

    file = download_file(URL, output.parent / "6S.tar")

    untar(file, output)

    print("Building via make")
    build(output)

    print(f"Done, now available at: {output}")


@cli.download.command(name="sixs")
@cli.output(help="Root directory to download sixs to, ie. [path]/sixs")
def download_cli(**kwargs):
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


def validate(path=None):
    """
    Validates a 6S installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if path is None:
        path = env.sixs

    print(f"Verifying path for 6S: {path}")

    if not (path := Path(path)).exists():
        print("Error: Path does not exist, please download it via `isofit download 6S`")
        return False

    if not (path / f"sixsV2.1").exists():
        print(
            "Error: 6S does not appear to be installed correctly, please ensure it is"
        )
        return False

    print("Path is valid")
    return True


@cli.validate.command(name="sixs")
@cli.path(help="Path to 6S installation")
def validate_cli(**kwargs):
    """\
    Validates a 6S installation
    """
    validate(**kwargs)
