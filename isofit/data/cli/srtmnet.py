"""
Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/
"""

import re
from pathlib import Path

import click
import requests
from packaging.version import Version

from isofit.data import env
from isofit.data.download import cli, download_file, prepare_output

URL = "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/"


def getVersion(version="latest"):
    """
    Retrieves the available versions and verifies the requested version is valid

    Parameters
    ----------
    version : str, default="latest"
        Version of sRTMnet to pull

    Returns
    -------
    str
        Requested version
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


def download(path=None, tag="latest", overwrite=False):
    """
    Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/.

    Parameters
    ----------
    output: str | None
        Path to output as. If None, defaults to the ini path
    tag: str
        sRTMnet version to pull
    overwrite : bool, default=False
        Overwrite an existing installation
    """
    if (version := getVersion(tag)) is None:
        return

    print(f"Downloading sRTMnet[{version}]")

    output = prepare_output(path, env.srtmnet, isdir=True, overwrite=overwrite)
    if not output:
        return

    print(f"Pulling version {version}")

    print("Retrieving model.h5")
    file = f"sRTMnet_{version}.h5"
    download_file(f"{URL}/{file}", output / file)

    env.changeKey("srtmnet.file", file)

    print("Retrieving aux.npz")
    file = f"sRTMnet_{version}_aux.npz"
    download_file(f"{URL}/{file}", output / file)

    env.changeKey("srtmnet.aux", file)
    env.save(diff_only=True)

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


def compare(file, name, version, error=print):
    """
    Compares an existing sRTMnet file against some other provided version

    Parameters
    ----------
    file : str
        sRTMnet model or aux file name
    name : str
        File type name for reporting with
    version : packaging.version.Version, default=None
        Version to compare against. If not given, retrieves the latest version from the server
    error : function, default=print
        Print function to use for error messages, eg. logging.error

    Returns
    -------
    bool
        True if the current version is not the latest, False otherwise
    """
    if version is None:
        version = Version(getVersion("latest"))

    if find := re.findall(r"(v\d+)", file):
        current = Version(find[0])
    else:
        error(f"Version could not be parsed from the path for {name}")
        return True

    if current < version:
        error(
            f"The sRTMnet {name} is out of date. The latest is v{version}, currently installed is v{current}"
        )
        return True

    return False


def checkForUpdate(path=None, tag="latest", debug=print, error=print, **_):
    """
    Checks the installed version against the latest release

    Parameters
    ----------
    path : str, default=None
        Path to update. If None, defaults to the ini path
    debug : function, default=print
        Print function to use for debug messages, eg. logging.debug
    error : function, default=print
        Print function to use for error messages, eg. logging.error
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions

    Returns
    -------
    bool
        True if there is a version update, else False
    """
    if not env["srtmnet.file"]:
        error(
            "sRTMnet file is not set in the ini, version unknown. Please either redownload or set the key 'srtmnet.file' in the ini"
        )
        return True

    if not env["srtmnet.aux"]:
        error(
            "sRTMnet aux is not set in the ini, version unknown. Please either redownload or set the key 'srtmnet.aux' in the ini"
        )
        return True

    if path is None:
        # If the paths do not exist, env.path() will report that
        model = env.path("srtmnet", key="srtmnet.file")
        aux = env.path("srtmnet", key="srtmnet.aux")

        if not model.exists() or not aux.exists():
            error("Download the above via `isofit download sRTMnet --update`")
            return True
    else:
        path = Path(path)
        model = path / env["srtmnet.file"]
        aux = path / env["srtmnet.aux"]

    debug(f"Checking for updates for sRTMnet on path: {model}")

    latest = Version(getVersion(tag))

    if compare(model.name, "model", latest, error) or compare(
        aux.name, "aux", latest, error
    ):
        error("Please update via `isofit download sRTMnet --update`")
        return True

    debug("Path is up to date")

    return False


def update(check=False, **kwargs):
    """
    Checks for an update and executes a new download if it is needed

    Parameters
    ----------
    check : bool, default=False
        Just check if an update is available, do not download
    **kwargs : dict
        Additional key-word arguments to pass to download()
    """
    kwargs["overwrite"] = True
    if checkForUpdate(**kwargs) and not check:
        download(**kwargs)


@cli.download.command(name="sRTMnet")
@cli.path(help="Root directory to download sRTMnet to, ie. [path]/sRTMnet")
@cli.tag
@cli.overwrite
@cli.update
@cli.check
@cli.validate
def download_cli(update_, check, validate_, **kwargs):
    """\
    Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --srtmnet /path/sRTMnet download sRTMnet`: Override the ini file. This will save the provided path for future reference.
        - `isofit download sRTMnet --output /path/sRTMnet`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if update_:
        update(check, **kwargs)
    elif validate_:
        validate(**kwargs)
    else:
        download(**kwargs)
