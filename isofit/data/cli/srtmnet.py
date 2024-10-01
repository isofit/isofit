"""
Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/
"""

import re

import click
import requests

from isofit.data import env
from isofit.data.download import cli_download, cli_opts, download_file, prepare_output

URL = "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/"


def getVersion(version="latest"):
    """
    Retrieves the available versions and verifies the requested version is valid
    """
    get = requests.get(URL)
    versions = list(set(re.findall(r"sRTMnet_(v\d+)\.h5", get.text)))
    versions = sorted(versions, key=lambda v: int(v[1:]))

    if version == "latest":
        return versions[-1]
    elif version in versions:
        return version
    else:
        click.echo(
            f"Error: Requested version {version!r} does not exist, must be one of: {versions}"
        )


def download(output=None, version="latest"):
    """
    Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/.

    Parameters
    ----------
    output: str | None
        Path to output as. If None, defaults to the ini path.
    version: str
        Release tag to pull from the github.
    """
    if (version := getVersion(version)) is None:
        return

    click.echo(f"Downloading sRTMnet[{version}]")

    output = prepare_output(output, env.srtmnet, isdir=True)
    if not output:
        return

    click.echo(f"Pulling version {version}")

    click.echo("Retrieving model.h5")
    file = f"sRTMnet_{version}.h5"
    download_file(f"{URL}/{file}", output / file)

    click.echo("Retrieving aux.npz")
    file = f"sRTMnet_{version}_aux.npz"
    download_file(f"{URL}/{file}", output / file)

    click.echo(f"Done, now available at: {output}")


@cli_download.command(name="sRTMnet")
@cli_opts.output(help="Root directory to download sRTMnet to, ie. [path]/sRTMnet")
@click.option(
    "-v",
    "--version",
    default="latest",
    help="Model version to download",
    show_default=True,
)
def cli_examples(**kwargs):
    """\
    Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --srtmnet /path/sRTMnet download sRTMnet`: Override the ini file. This will save the provided path for future reference.
        - `isofit download sRTMnet --output /path/sRTMnet`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download(**kwargs)
