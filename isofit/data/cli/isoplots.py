"""
Downloads the extra ISOFIT plotting utilities from the repository https://github.com/isofit/isofit-plots
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output, release_metadata, unzip

CMD = "plots"


def install(path=None):
    """
    Installs a python package into the current python environment as editable mode

    Parameters
    ----------
    path : str | None
        Path to package to install. If None, defaults to the ini path
    """
    if path is None:
        path = env.plots

    print(f"Installing {path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", path])


def download(path=None, tag="latest", overwrite=False):
    """
    Downloads the extra ISOFIT plotting utilities from the repository https://github.com/isofit/isofit-plots.

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path
    tag : str
        Release tag to pull from the github
    overwrite : bool, default=False
        Overwrite an existing installation
    """
    print(f"Downloading ISOFIT Plotting Utilities")

    output = prepare_output(path, env.plots, overwrite=overwrite)
    if not output:
        return

    metadata = release_metadata("isofit", "isofit-plots", tag)

    print(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(metadata["zipball_url"], output.parent / "isofit-plots.zip")

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name, overwrite=overwrite)

    install(output)

    print(f"Done, now available at: {avail}")


def validate(path=None, checkForUpdate=True, debug=print, error=print, **_):
    """
    Validates an isoplots installation

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
        path = env.plots

    debug(f"Verifying path for isoplots: {path}")

    # Copy the existing sys.path
    syspath = sys.path[:]
    if Path(path).exists():
        sys.path.append(path)

    try:
        import isoplots
    except Exception as e:
        error(f"[x] Failed to load isoplots: {e}")
        return False
    finally:
        # Remove the inserted path
        sys.path = syspath

    if checkForUpdate:
        return isUpToDate(path, debug=debug, error=error)

    debug("[OK] Path is valid")
    return True


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
    _ : dict
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
    if path is None:
        path = env.plots

    debug(f"Checking for updates for plots on path: {path}")

    latest = Version(release_metadata("isofit", "isofit-plots", "latest")["tag_name"])
    current = Version(importlib.metadata.version("isoplots"))

    if current < latest:
        error(f"[x] Latest is {latest}, currently installed is {version}")
        return False

    debug(f"[OK] Path is up to date, current version: {current}")

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
@shared.path(
    help="Root directory to download the ISOFIT plotting utilities package to, ie. [path]/plots"
)
@shared.tag
@shared.overwrite
@shared.check
def download_cli(**kwargs):
    """\
    Downloads the extra ISOFIT plotting utilities from the repository https://github.com/isofit/isofit-plots.
    This will install the package into your current environment along with its dependencies.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --plots /path/plots download plots`: Override the ini file. This will save the provided path for future reference.
        - `isofit download plots --path /path/plots`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Root directory to download data files to, ie. [path]/data")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of the ISOFIT extra data files as well as checks for updates
    """
    validate(**kwargs)
