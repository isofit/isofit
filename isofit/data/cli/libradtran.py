"""
Downloads LibRadTran from https://www.libradtran.org/download/libRadtran-2.0.6.tar.gz
"""

import os
import platform
import subprocess
from pathlib import Path

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output, untar

Version = "libRadtran-2.0.6"
CMD = "libradtran"
URL = f"https://www.libradtran.org/download/{Version}.tar.gz"


def build(directory):
    """
    Builds a LibRadTran directory

    Parameters
    ----------
    directory : str
        Directory with an unbuilt LibRadTran

    Notes
    -----
    If on MacOS, executing the `make` command may fail if the user hasn't agreed to the
    Xcode and Apple SDKs license yet. In these cases, it may be required to run the
    following command in order to compile the program:
    $ sudo xcodebuild -license
    """
    flags = []

    # LibRadTran assumes GSL was installed with MacPorts
    # Check if it was installed via brew instead and fix paths if so

    if Path("/opt/homebrew/Cellar/gsl").exists():
        flags = [
            'CPPFLAGS="-I/opt/homebrew/include"',
            'LDFLAGS="-L/opt/homebrew/lib"',
            'FCFLAGS="-I/opt/homebrew/include"',
            'FFLAGS="-I/opt/homebrew/include"',
            "--with-netcdf4=/opt/homebrew",
        ]

    # Configure first
    command = f"./configure {' '.join(flags)}"
    print(f"Executing: {command}")
    subprocess.run(
        command,
        shell=True,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        cwd=directory,
    )

    # Now make it
    print("Executing make")
    subprocess.run(
        "make",
        shell=True,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        cwd=directory,
    )


def download(path=None, overwrite=False, **_):
    """
    Downloads LibRadTran from https://www.libradtran.org/download/libRadtran-2.0.6.tar.gz.

    Parameters
    ----------
    output : str | None
        Path to output as. If None, defaults to the ini path.
    overwrite : bool, default=False
        Overwrite an existing installation
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions
    """
    print("Downloading LibRadTran")

    output = prepare_output(path, env.libradtran, overwrite=overwrite)
    if not output:
        return

    file = download_file(URL, output.parent / "LibRadTran.tar")

    untar(file, output)

    print("Building via make")
    build(output / Version)

    env.changeKey("libradtran.version", Version)
    env.save()

    print(f"Done, now available at: {output}")


def validate(path=None, debug=print, error=print, **_):
    """
    Validates a LibRadTran installation

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
        path = env.libradtran

    debug(f"Verifying path for LibRadTran: {path}")

    if not (path := Path(path)).exists():
        error("[x] LibRadTran path does not exist")
        return False

    debug("[OK] Path is valid")
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
    debug = kwargs.get("debug", print)
    if not validate(**kwargs):
        if not check:
            kwargs["overwrite"] = True
            debug("Executing update")
            download(**kwargs)
        else:
            debug(f"Please download the latest via `isofit download {CMD}`")


@shared.download.command(name=CMD)
@shared.path(help="Root directory to download LibRadTran to, ie. [path]/libradtran")
@shared.tag
@shared.overwrite
@shared.check
def download_cli(**kwargs):
    """\
    Downloads LibRadTran from https://www.libradtran.org/download/libRadtran-2.0.6.tar.gz. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --path libradtran /path/libradtran download libradtran`: Override the ini file. This will save the provided path for future reference.
        - `isofit download libradtran --path /path/libradtran`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Root directory to download LibRadTran to, ie. [path]/libradtran")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of LibRadTran
    """
    validate(**kwargs)
