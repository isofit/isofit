"""
Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/
"""

import re
from pathlib import Path

import click
import requests
from packaging.version import Version

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output

CMD = "srtmnet"
URL = "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/"


def getVersion(version="latest"):
    """
    Retrieves the available versions and verifies the requested version is valid. Times
    out after 10 seconds if the server is unavailable.

    Parameters
    ----------
    version : str, default="latest"
        Version of sRTMnet to pull

    Returns
    -------
    str
        Requested version
    """
    try:
        get = requests.get(URL, timeout=10)
    except requests.exceptions.Timeout:
        print("[!] sRTMnet server request timed out, cannot retrieve versions")
        return
    except requests.exceptions.ConnectionError:
        print("[!] sRTMnet server refused connection, cannot retrieve versions")
        return

    versions = list(set(re.findall(r"sRTMnet_(v\d+)\.h5", get.text)))
    versions = sorted(versions, key=lambda v: int(v[1:]))

    if version == "latest":
        return versions[-1]
    elif version in versions:
        return version
    else:
        print(
            f"[!] Requested version {version!r} does not exist, must be one of: {versions}"
        )


def download(path=None, tag="latest", overwrite=False, **_):
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
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions
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


def validate(path=None, checkForUpdate=True, debug=print, error=print, **_):
    """
    Validates an sRTMnet installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path
    checkForUpdate : bool, default=True
        Checks for updates if the path is valid
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
        error("[x] sRTMnet path does not exist")
        return False

    if not list(path.glob("*.h5")):
        error("[x] sRTMnet model not found")
        return False

    if not list(path.glob("*_aux.npz")):
        error("[x] sRTMnet aux file not found")
        return False

    debug("[OK] Path is valid")

    if checkForUpdate:
        isUpToDate(path, debug=debug, error=error)

    return True


def detectInstalled(path: str = None):
    """
    Attempt to detect a currently installed sRTMnet model and, if present, save that
    information back to the ini for future reference

    Parameters
    ----------
    path : str, default=None
        Path to check. If None, defaults to the ini path
    """
    path = None

    if path is None:
        path = env.srtmnet

    path = Path(path)

    if path.exists():
        try:
            # Just retrieve the first of each extension
            # This may be inaccurate if multiple installs are in one directory
            env.changeKey("srtmnet.file", next(path.glob("*.h5")).name)
            env.changeKey("srtmnet.aux", next(path.glob("*.npz")).name)
            env.save()
        except:
            pass


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
        Version to compare against. If not given, retrieves the latest version from the
        server
    error : function, default=print
        Print function to use for error messages, eg. logging.error

    Returns
    -------
    bool
        True if the current version is not the latest, False otherwise
    """
    if version is None:
        if not (latest := getVersion(tag)):
            error("[!] Failed to retrieve latest version, try again later")
            return False

        latest = Version(latest)

    if find := re.findall(r"(v\d+)", file):
        current = Version(find[0])
    else:
        error(f"[x] Version could not be parsed from the path for {name}")
        return True

    if current < version:
        error(
            f"[x] The sRTMnet {name} is out of date. The latest is v{version}, currently installed is v{current}"
        )
        return True

    return False


def isUpToDate(path=None, tag="latest", debug=print, error=print, **_):
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
        True if the path is up to date, False otherwise

    Notes
    -----
    The Github workflows watch for the string "[x]" to determine if the cache needs to
    update the data of this module. If your module does not include this string, the
    workflows will never detect updates.
    """
    missing = False
    if not env["srtmnet.file"]:
        error("[x] sRTMnet file is not set in the ini, version unknown")
        missing = True

    if not env["srtmnet.aux"]:
        error("[x] sRTMnet aux is not set in the ini, version unknown")
        missing = True

    if missing:
        debug(
            "Attempting to detect if a model and aux are already installed but not set in the ini"
        )
        detectInstalled(path)

    if path is None:
        # If the paths do not exist, env.path() will report that
        model = env.path("srtmnet", key="srtmnet.file")
        aux = env.path("srtmnet", key="srtmnet.aux")

        if not model.exists() or not aux.exists():
            error("[x] Missing the above")
            return False
    else:
        path = Path(path)
        model = path / env["srtmnet.file"]
        aux = path / env["srtmnet.aux"]

    debug(f"Checking for updates for sRTMnet on path: {model}")

    if not (latest := getVersion(tag)):
        error("[!] Failed to retrieve latest version, try again later")
        return True

    latest = Version(latest)
    model = compare(model.name, "model", latest, error)
    aux = compare(aux.name, "aux", latest, error)

    if model or aux:
        return False

    debug("[OK] Path is up to date")

    return True


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
    debug = kwargs.get("debug", print)
    if not validate(**kwargs):
        if not check:
            kwargs["overwrite"] = True
            debug("Executing update")
            download(**kwargs)
        else:
            debug(f"Please download the latest via `isofit download {CMD}`")


@shared.download.command(name=CMD)
@shared.path(help="Root directory to download sRTMnet to, ie. [path]/sRTMnet")
@shared.tag
@shared.overwrite
@shared.check
def download_cli(**kwargs):
    """\
    Downloads sRTMnet from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --srtmnet /path/sRTMnet download sRTMnet`: Override the ini file. This will save the provided path for future reference.
        - `isofit download sRTMnet --path /path/sRTMnet`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Root directory to download sRTMnet to, ie. [path]/sRTMnet")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of sRTMnet as well as checks for updates
    """
    validate(**kwargs)
