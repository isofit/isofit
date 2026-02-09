"""
Downloads the extra ISOFIT surface prior files from https://github.com/emit-sds/emit-sds-l2a/tree/main/surface
"""

import importlib.metadata
import shutil
from pathlib import Path

import click
from packaging.version import Version

from isofit.data import env, shared
from isofit.data.download import (
    isUpToDateGithub,
    prepare_output,
    pullFromRepo,
    release_metadata,
)

ESSENTIAL = False
CMD = "surface"


def download(path=None, tag="latest", overwrite=False, **_):
    """
    Downloads the ISOFIT surface prior files from the repository https://github.com/emit-sds/emit-sds-l2a/tree/main/surface.

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path
    tag : str
        Release tag to pull from the github
    overwrite : bool, default=False
        Overwrite an existing installation
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions
    """
    print("Downloading ISOFIT surface")

    output = prepare_output(path, env.surface, overwrite=overwrite)
    if not output:
        return

    avail = pullFromRepo(
        owner="emit-sds",
        repo="emit-sds-l2a",
        tag=tag,
        output=output.parent / "surface_repository",
        overwrite=overwrite,
    )
    shutil.move(avail / "surface", avail / "..")
    shutil.move(avail / "version.txt", avail / ".." / "surface" / "version.txt")
    shutil.rmtree(avail)


def validate(path=None, checkForUpdate=True, debug=print, error=print, **_):
    """
    Validates an ISOFIT surface installation

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
        path = env.surface

    debug(f"Verifying path for ISOFIT surface: {path}")

    path = Path(path)

    expected = set(
        [
            "filtered_ocean",
            "filtered_other",
            "filtered_veg",
            "surface_Liquids",
            "surface_mixture_veg_soil",
            "surface_SWIPE",
        ]
    )
    names = set([file.name for file in path.glob("*")])
    if missing := (expected - names):
        error("[x] ISOFIT surface is missing surface library files")
        debug(f"Expected: {expected}")
        debug(f"Got: {names}")
        debug(f"Missing: {missing}")
        return False

    debug("[OK] Path is valid")

    if checkForUpdate:
        return isUpToDateGithub(
            owner="emit-sds", repo="emit-sds-l2a", name="surface", path=path
        )

    debug("[OK] Path is valid")
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
@shared.path(help="Root directory to download surface files to, ie. [path]/surface")
@shared.tag
@shared.overwrite
@shared.check
def download_cli(**kwargs):
    """\
    Downloads the extra ISOFIT surface prior files from https://github.com/emit-sds/emit-sds-l2a/tree/main/surface

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --path surface /path/surface download surface`: Override the ini file. This will save the provided path for future reference.
        - `isofit download surface --path /path/surface`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Root directory to download example files to, ie. [path]/surface")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of the ISOFIT surface as well as checks for updates
    """
    validate(**kwargs)
