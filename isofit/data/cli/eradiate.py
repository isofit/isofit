"""
Downloads the Eradiate data assets from the Eradiate data registry https://github.com/eradiate/eradiate
"""

import importlib.metadata
import logging
import os
from pathlib import Path

import click

from isofit.data import env, shared
from isofit.data.download import prepare_output

Logger = logging.getLogger(__name__)

ESSENTIAL = False
CMD = "eradiate"

# Eradiate resource alias(es) to install, see eradiate/data/resources.yml
# The "core" alias includes the solar irradiance, SRF, and aerosol data, along
# with the komodo and monotropa absorption databases. panellus is the absorption
# database apply_oe configures by default, but is not part of "core"
RESOURCES = ["core", "panellus"]


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


def set_data_path(path=None):
    """
    Points Eradiate's asset manager and file resolver at the ini eradiate path.
    Eradiate otherwise takes its data path from the ERADIATE_DATA_PATH
    environment variable when it is first imported, so this is applied after
    import, and must be called in every process that uses Eradiate (eg. ray
    workers)

    Parameters
    ----------
    path : str | None
        Eradiate data path. If None, defaults to the ini path

    Returns
    -------
    pathlib.Path
        The data path applied
    """
    from eradiate import asset_manager, fresolver
    from eradiate.config import settings

    path = Path(path or env.eradiate).expanduser().resolve()

    current = os.environ.get("ERADIATE_DATA_PATH")
    if current and Path(current).expanduser().resolve() != path:
        Logger.warning(
            "The environment variable $ERADIATE_DATA_PATH does not match the ISOFIT ini, the INI path will be the one used"
        )
        Logger.warning(f"ENV: {current}")
        Logger.warning(f"INI: {path}")

    # install_dir is [data_path]/installed/eradiate-v[version]
    old = asset_manager.install_dir
    if old.parents[1] == path:
        return path

    settings.set("DATA_PATH", str(path))
    asset_manager.cache_dir = path / "cached"
    asset_manager.unpack_dir = path / "unpacked"
    asset_manager.install_dir = path / "installed" / old.name

    # Creates the directories and reloads the manifest and installed resources
    asset_manager.__attrs_post_init__()

    # Swap the old install location in place to keep the search order
    if old in fresolver.paths:
        fresolver.paths[fresolver.paths.index(old)] = asset_manager.install_dir
    else:
        fresolver.append(asset_manager.install_dir)

    Logger.debug(f"Set the Eradiate data path to: {path}")

    return path


def check_databases(databases, path=None):
    """
    Checks that Eradiate absorption databases are installed under the ini
    eradiate path

    Parameters
    ----------
    databases : list[str]
        Absorption database names, eg. ["panellus"]
    path : str | None
        Eradiate data path. If None, defaults to the ini path

    Returns
    -------
    list[str]
        Error messages, one per database that is unknown or not installed. Empty
        if all are installed

    Raises
    ------
    ImportError
        If Eradiate is not installed
    """
    from eradiate import fresolver
    from eradiate.radprops._absorption import KNOWN_DATABASES

    path = set_data_path(path)

    errors = []
    for database in databases:
        if database not in KNOWN_DATABASES:
            errors.append(
                f"Unknown Eradiate absorption database {database!r}, must be one of: "
                f"{', '.join(KNOWN_DATABASES)}"
            )
            continue

        try:
            found = fresolver.resolve(
                KNOWN_DATABASES[database]["path"], strict=True
            ).is_dir()
        except FileNotFoundError:
            found = False

        if not found:
            errors.append(
                f"Eradiate absorption database {database!r} is not installed under "
                f"{path}. Install it via: isofit download {CMD} --database {database}"
            )

    return errors


def install_databases(databases, path=None):
    """
    Installs additional Eradiate absorption databases into an existing
    installation, without redownloading the rest

    Parameters
    ----------
    databases : list[str]
        Absorption database names, eg. ["tuber"]
    path : str | None
        Eradiate data path. If None, defaults to the ini path
    """
    from eradiate import asset_manager

    path = set_data_path(path)

    print(f"Installing Eradiate absorption database(s) {list(databases)} to: {path}")
    asset_manager.install(list(databases), progressbar=True)
    print("Done")


def download(path=None, overwrite=False, database=(), **_):
    """
    Downloads the Eradiate data assets from the Eradiate data registry.

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path
    overwrite : bool, default=False
        Overwrite an existing installation
    database : list[str], default=()
        Additional absorption databases to install alongside RESOURCES
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

    try:
        from eradiate import asset_manager
    except ImportError as e:
        # Eradiate may be present without its kernel (mitsuba/drjit)
        print(f"Failed to import Eradiate: {e}")
        print_install()
        return

    set_data_path(output)
    asset_manager.install(RESOURCES + list(database), progressbar=True)

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
@click.option(
    "-d",
    "--database",
    multiple=True,
    help="Additional absorption database to install, eg. tuber. May be repeated",
)
def download_cli(**kwargs):
    """\
    Downloads the Eradiate data assets from the Eradiate data registry. Requires the eradiate package to be installed.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --path eradiate /path/eradiate download eradiate`: Override the ini file. This will save the provided path for future reference.
        - `isofit download eradiate --path /path/eradiate`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.

    \b
    Additional absorption databases can be added to an existing installation via:
        - `isofit download eradiate --database tuber`
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    elif kwargs.get("database") and validate(checkForUpdate=False, **kwargs):
        install_databases(kwargs["database"], path=kwargs.get("path"))
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
