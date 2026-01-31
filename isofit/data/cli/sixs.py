"""
Downloads 6S from https://github.com/isofit/6S
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

import click

from isofit.data import env, shared
from isofit.data.download import (
    download_file,
    isUpToDateGithub,
    prepare_output,
    pullFromRepo,
    unzip,
)

ESSENTIAL = True
CMD = "sixs"
MINGW = "https://github.com/brechtsanders/winlibs_mingw/releases/download/15.2.0posix-13.0.0-msvcrt-r2/winlibs-i686-posix-dwarf-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r2.zip"

Logger = logging.getLogger(__name__)


def get_exe(path: str = None, version: bool = False) -> str:
    """
    Retrieves the 6S executable from a given path

    Parameters
    ----------
    path : str, default=None
        6S directory path. If None, defaults to the ini sixs path
    version : bool, default=False
        Returns the 6S version instead

    Returns
    -------
    pathlib.Path | str
        Either the 6S executable as a pathlib object or the string 6S version
    """
    if path is None:
        path = env.sixs

    path = Path(path)

    exes = path.glob("sixsV*")
    exes = [exe for exe in exes if "lutaero" not in exe.name]
    names = [exe.name for exe in exes]

    if not exes:
        raise FileNotFoundError(f"Could not find a 6S executable under path: {path}")

    if len(exes) > 1:
        Logger.warning(
            f"More than one 6S executable was found. Defaulting to the first one: {names}"
        )

    if version:
        # Try using the version.txt file, created by the isofit downloader
        if (txt := path / "version.txt").exists():
            with txt.open("r") as f:
                vers = f.read()
        # Fallback to using the executable name
        else:
            _, vers = names[0].split("V")
            vers = f"v{vers}".lower()

        return vers

    return exes[0]


def precheck():
    """
    Checks if gfortran is installed before downloading SixS

    Returns
    -------
    True or None
        True if `gfortran --version` returns a valid response, None otherwise
    """
    proc = subprocess.run("gfortran --version", shell=True, stdout=subprocess.PIPE)

    if proc.returncode == 0:
        return True

    print(
        f"Failed to validate an existing gfortran installation. Please ensure it is installed on your system."
    )


def patch_makefile(file):
    """
    Patch the 6S Makefile to:
    - Add -std=legacy to the EXTRAS (isofit)

    Parameters
    ----------
    file : pathlib.Path
        Makefile to patch inplace
    """
    lines = file.read_text().splitlines()

    # Insert new lines
    flags = [
        "EXTRA   = -O -ffixed-line-length-132 -std=legacy",
    ]
    for i, flag in enumerate(flags, start=3):
        if lines[i] != flag:
            lines.insert(i, flag)

    file.write_text("\n".join(lines))


def make(directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, debug=False):
    """
    Builds a 6S directory via make

    Parameters
    ----------
    directory : str
        6S directory to build

    Notes
    -----
    If on MacOS, executing the `make` command may fail if the user hasn't agreed to the
    Xcode and Apple SDKs license yet. In these cases, it may be required to run the
    following command in order to compile the program:
    $ sudo xcodebuild -license
    """
    # Update the makefile with recommended flags
    file = Path(directory) / "Makefile"
    patch_makefile(file)

    make = "make"
    if platform.system() == "Windows":
        make = "mingw32-make.exe"

        proc = subprocess.run(f"{make} --help", shell=True, stdout=subprocess.PIPE)
        if proc.returncode != 0:
            print("MinGW64 not found, downloading")
            download_mingw()

    kwargs = dict(
        shell=True,
        check=True,
        stdout=stdout,
        stderr=stderr,
        cwd=directory,
    )

    try:
        # Clean *.o files
        call = subprocess.run(f"{make} clean", **kwargs)

        # Build sixs
        call = subprocess.run(f"{make} -j {os.cpu_count()}", **kwargs)
        if debug:
            if call.stdout:
                print(f"stdout " + "-" * 32 + f"\n{call.stdout.decode()}")
            if call.stderr:
                print(f"stderr " + "-" * 32 + f"\n{call.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Building 6S via make failed, exit code: {e.returncode}")
        print(e.stderr)


def download_mingw(path=None, tag="latest", overwrite=False, **_):
    """
    Downloads MinGW64 for Windows

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
    print("Downloading MinGW64")

    output = prepare_output(path, env.path("sixs", "MinGW64"), overwrite=overwrite)
    if not output:
        return

    zipfile = download_file(MINGW, output.parent / "MinGW64.zip")
    avail = unzip(zipfile, path=output.parent, rename=output.name, overwrite=overwrite)

    print(f"Done, now available at: {avail}/bin")
    print(
        "You may need to add it to your PATH environment variable, ISOFIT will also do this automatically at runtime"
    )

    env.changeKey("path.mingw", "{sixs}/MinGW64/bin")
    env.save()
    env.load()  # Reload to insert the MinGW64 path to $PATH


def download(path=None, tag="latest", overwrite=False, debug_make=False, **_):
    """
    Downloads 6S from https://github.com/isofit/6S.

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
    if not precheck():
        print(
            "Skipping downloading 6S. Once above errors are corrected, retry via `isofit download sixs`"
        )
        return

    print("Downloading 6S")

    output = prepare_output(path, env.sixs, overwrite=overwrite)
    if not output:
        return

    avail = pullFromRepo("isofit", "6S", tag, output, overwrite=overwrite)

    # Move files from subdir to base dir for backwards compatibility
    for path in (avail / "Sixs").iterdir():
        shutil.move(path, avail / path.name)

    print("Building via make")
    make(avail, debug=debug_make)

    print(f"Done, now available at: {avail}")


def validate(path=None, checkForUpdate=True, debug=print, error=print, **_):
    """
    Validates a 6S installation

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
        path = env.sixs

    debug(f"Verifying path for 6S: {path}")

    path = Path(path)

    if not path.exists():
        error("[x] 6S path does not exist")
        return False

    try:
        exe = get_exe(path)
    except FileNotFoundError:
        error(
            "[x] 6S is missing the built 'sixsV2.*', this is likely caused by make failing"
        )
        return False

    if checkForUpdate:
        return isUpToDateGithub(owner="isofit", repo="6S", name="sixs", path=path)

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
@shared.path(help="Root directory to download 6S to, ie. [path]/sixs")
@shared.tag
@shared.overwrite
@shared.check
@click.option("--make", is_flag=True, help="Builds a 6S directory via make")
@click.option(
    "--debug-make", is_flag=True, help="Enable debug logging for the make command"
)
@click.option(
    "--mingw",
    is_flag=True,
    help="Downloads the MinGW64 (for Windows) instead of 6S",
)
def download_cli(debug_make, mingw, **kwargs):
    """\
    Downloads 6S from https://github.com/isofit/6S. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --path sixs /path/sixs download sixs`: Override the ini file. This will save the provided path for future reference.
        - `isofit download sixs --path /path/sixs`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("make"):
        path = kwargs.get("path")
        if path is None:
            path = env.sixs

        print(f"Making 6S: {path}")
        make(path, debug=debug_make)
        print(f"Finished")
    elif kwargs.get("overwrite"):
        download(**kwargs)
    elif mingw:
        download_mingw(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Root directory to download 6S to, ie. [path]/sixs")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of 6S
    """
    validate(**kwargs)
