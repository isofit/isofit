"""
Downloads MODTRAN
"""

from pathlib import Path

from isofit.data import env
from isofit.data.download import (
    cli,
    download_file,
    prepare_output,
    release_metadata,
    unzip,
)


def download(output=None):
    """
    TODO

    Parameters
    ----------
    output: str | None
        Path to output as. If None, defaults to the ini path.
    """
    print("MODTRAN downloading is not supported yet")


# @cli.download.command(name="modtran")
# @cli.output(help="Root directory to download modtran files to, ie. [path]/modtran")
# @cli.tag
def download_cli(**kwargs):
    """\
    Downloads and installs MODTRAN

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --modtran /path/modtran download modtran`: Override the ini file. This will save the provided path for future reference.
        - `isofit download modtran --output /path/modtran`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download(**kwargs)


def validate(path=None, debug=print, error=print, **_):
    """
    Validates an ISOFIT data installation

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
    # TODO: Write a proper validation function
    return True

    if path is None:
        path = env.modtran

    debug(f"Verifying path for MODTRAN: {path}")

    ...

    debug("Path is valid")
    return True


# @cli.validate.command(name="modtran")
# @cli.path(help="Path to a MODTRAN installation")
def validate_cli(**kwargs):
    """\
    Validates a MODTRAN installation
    """
    validate(**kwargs)
