"""
Downloads the Eradiate data assets from the Eradiate data registry https://github.com/eradiate/eradiate
"""

import importlib.metadata
import os
import sys
from pathlib import Path

from isofit.data import env, shared
from isofit.data.download import prepare_output

ESSENTIAL = False
CMD = "eradiate"

# Eradiate resource alias(es) to install, see eradiate/data/resources.yml
# The "core" alias includes the default absorption database (komodo) used by the
# ISOFIT eradiate engine, along with the solar irradiance, SRF, and aerosol data
RESOURCES = ["core"]


def print_install():
    """
    Prints the recommended command to install the optional eradiate dependencies
    """
    print("Eradiate is not installed, it is an optional dependency of ISOFIT")
    print('To install it, run: pip install "isofit[eradiate]"')
    print("Or if using uv: uv sync --extra eradiate")


def installed_version():
    """
    Retrieves the installed Eradiate package version

    Returns
    -------
    str | None
        The installed Eradiate version, None if Eradiate is not installed
    """
    try:
        return importlib.metadata.version("eradiate")
    except importlib.metadata.PackageNotFoundError:
        return None


def download(path=None, overwrite=False, **_):
    """
    Downloads the Eradiate data assets from the Eradiate data registry.

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path
    overwrite : bool, default=False
        Overwrite an existing installation
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions
    """
    # Check before preparing the output so an empty directory isn't left behind
    if installed_version() is None:
        print("Skipping downloading Eradiate data")
        print_install()
        return

    print("Downloading Eradiate data")

    output = prepare_output(path, env.eradiate, isdir=True, overwrite=overwrite)
    if not output:
        return

    # Eradiate reads its data path when it is first imported, so this must be set first
    if "eradiate" in sys.modules:
        print(
            "[x] Eradiate was already imported in this process, the download path cannot be applied. Please retry in a new process"
        )
        return
    os.environ["ERADIATE_DATA_PATH"] = str(output)

    try:
        from eradiate import asset_manager
    except ImportError as e:
        # Eradiate may be present without its kernel (mitsuba/drjit)
        print(f"Failed to import Eradiate: {e}")
        print_install()
        return

    # install_dir is [data_path]/installed/eradiate-v[version]
    resolved = Path(asset_manager.install_dir).parents[1]
    if resolved.resolve() != output.resolve():
        print(f"[x] Eradiate resolved its data path to {resolved}, not {output}")
        return

    asset_manager.install(RESOURCES, progressbar=True)

    with open(output / "version.txt", "w") as file:
        file.write(installed_version() or "unknown")

    print(f"Done, now available at: {output}")


def validate(path=None, checkForUpdate=True, debug=print, error=print, **_):
    """
    Validates an Eradiate data installation

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
        path = env.eradiate

    debug(f"Verifying path for Eradiate: {path}")

    if not (path := Path(path)).exists():
        error("[x] Eradiate path does not exist")
        return False

    # Eradiate installs assets under [path]/installed/eradiate-v[version]
    if not any((path / "installed").glob("eradiate-v*")):
        error("[x] Eradiate data do not appear to be installed correctly")
        return False

    debug("[OK] Path is valid")

    if checkForUpdate:
        return isUpToDate(path, debug=debug, error=error)

    return True


def isUpToDate(path=None, debug=print, error=print, **_):
    """
    Checks the installed data version against the installed Eradiate package version

    Parameters
    ----------
    path : str, default=None
        Path to check. If None, defaults to the ini path
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
    if path is None:
        path = env.eradiate

    debug(f"Checking for updates for Eradiate data on path: {path}")

    file = Path(path) / "version.txt"
    if not file.exists():
        error(
            "[x] Failed to find a version.txt file under the given path. Version is unknown"
        )
        return False

    if (latest := installed_version()) is None:
        error("[x] Eradiate is not installed, unable to check for updates")
        return False

    with open(file, "r") as f:
        current = f.read()

    if current != latest:
        error(f"[x] Latest is {latest}, currently installed is {current}")
        return False

    debug(f"[OK] Path is up to date, current version is: {current}")

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
            if installed_version() is None:
                print_install()


@shared.download.command(name=CMD)
@shared.path(help="Root directory to download Eradiate data to, ie. [path]/eradiate")
@shared.tag
@shared.overwrite
@shared.check
def download_cli(**kwargs):
    """\
    Downloads the Eradiate data assets from the Eradiate data registry. Requires the eradiate package to be installed.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --path eradiate /path/eradiate download eradiate`: Override the ini file. This will save the provided path for future reference.
        - `isofit download eradiate --path /path/eradiate`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Root directory to download Eradiate data to, ie. [path]/eradiate")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of the Eradiate data assets
    """
    validate(**kwargs)
