"""
Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/
"""

import re
from pathlib import Path

import click
import requests

from isofit.data import env
from isofit.data.download import cli, download_file, prepare_output

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
        print(
            f"Error: Requested version {version!r} does not exist, must be one of: {versions}"
        )


def download(path=None, version="latest"):
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

    print(f"Downloading sRTMnet[{version}]")

    output = prepare_output(path, env.srtmnet, isdir=True)
    if not output:
        return

    print(f"Pulling version {version}")

    print("Retrieving model.h5")
    file = f"sRTMnet_{version}.h5"
    download_file(f"{URL}/{file}", output / file)

    print("Retrieving aux.npz")
    file = f"sRTMnet_{version}_aux.npz"
    download_file(f"{URL}/{file}", output / file)

    print(f"Done, now available at: {output}")


def validate(path=None, debug=print, error=print, **_):
    """
    Validates an sRTMnet installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path
    debug : function, default=print
        Print function to use for debug messages, eg. logging.debug
    error : function, default=print
        Print function to use for error messages, eg. logging.error
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with env.validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if path is None:
        path = env.srtmnet

    debug(f"Verifying path for sRTMnet: {path}")

    if not (path := Path(path)).exists():
        error(
            "Error: sRTMnet path does not exist, please download it via `isofit download sRTMnet`"
        )
        return False

    if not list(path.glob("*.h5")):
        error("Error: sRTMnet model not found, please download it")
        return False

    if not list(path.glob("*_aux.npz")):
        error("Error: sRTMnet aux file not found, please download it")
        return False

    debug("Path is valid")
    return True


def update(check=False, **kwargs):
    """
    Checks for an update and executes a new download if it is needed
    Note: Not implemented for this module at this time

    Parameters
    ----------
    check : bool, default=False
        Just check if an update is available, do not download
    **kwargs : dict
        Additional key-word arguments to pass to download()
    """
    # TODO: Implement, requires some changes to how downloading is handled
    pass


@cli.download.command(name="sRTMnet")
@cli.output(help="Root directory to download sRTMnet to, ie. [path]/sRTMnet")
@click.option(
    "-v",
    "--version",
    default="latest",
    help="Model version to download",
    show_default=True,
)
@cli.validate
def download_cli(validate_, **kwargs):
    """\
    Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --srtmnet /path/sRTMnet download sRTMnet`: Override the ini file. This will save the provided path for future reference.
        - `isofit download sRTMnet --output /path/sRTMnet`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if validate_:
        validate(**kwargs)
    else:
        download(**kwargs)
