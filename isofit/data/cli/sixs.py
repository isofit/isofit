"""
Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar
"""

import os
import platform
import subprocess
from pathlib import Path

import click

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output, untar, unzip

CMD = "sixs"
URL = "https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar"
MINGW = "https://github.com/brechtsanders/winlibs_mingw/releases/download/15.2.0posix-13.0.0-msvcrt-r2/winlibs-i686-posix-dwarf-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r2.zip"


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

    # if platform.system() == "Windows":
    #     for i, line in enumerate(lines):
    #         if "-lm" in line:
    #             lines[i].replace("-lm", "")

    file.write_text("\n".join(lines))


def make(directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, debug=False):
    """
    Builds a 6S directory via make

    Parameters
    ----------
    directory : str
        6S directory to build
    """
    # Update the makefile with recommended flags
    file = Path(directory) / "Makefile"
    patch_makefile(file)

    make = "make"
    if platform.system() == "Windows":
        make = "mingw32-make.exe"

    # Now make it
    try:
        call = subprocess.run(
            f"{make} -j {os.cpu_count()}",
            shell=True,
            check=True,
            stdout=stdout,
            stderr=stderr,
            cwd=directory,
        )
        if debug:
            if call.stdout:
                print(f"stdout " + "-" * 32 + f"\n{call.stdout.decode()}")
            if call.stderr:
                print(f"stderr " + "-" * 32 + f"\n{call.stderr.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Building 6S via make failed, exit code: {e.returncode}")
        print(e.stderr)


def download_mingw(path=None, overwrite=False, **_):
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


def download(path=None, overwrite=False, debug_make=False, **_):
    """
    Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar.

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

    file = download_file(URL, output.parent / "6S.tar")

    untar(file, output)

    print("Building via make")
    make(output, debug=debug_make)

    print(f"Done, now available at: {output}")


def validate(path=None, debug=print, error=print, **_):
    """
    Validates a 6S installation

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
        path = env.sixs

    debug(f"Verifying path for 6S: {path}")

    if not (path := Path(path)).exists():
        error("[x] 6S path does not exist")
        return False

    if not (path / f"sixsV2.1").exists():
        error(
            "[x] 6S is missing the built 'sixsV2.1', this is likely caused by make failing"
        )
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
    Downloads 6S from https://github.com/ashiklom/isofit/releases/download/6sv-mirror/6sv-2.1.tar. Only HDF5 versions are supported at this time.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --sixs /path/sixs download sixs`: Override the ini file. This will save the provided path for future reference.
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
