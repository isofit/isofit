"""
Windows helper downloader (MinGW64 and Portable Git)
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

import click

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output, unzip

ESSENTIAL = False
CMD = "windows"
MINGW = "https://github.com/brechtsanders/winlibs_mingw/releases/download/15.2.0posix-13.0.0-msvcrt-r2/winlibs-i686-posix-dwarf-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r2.zip"
GIT_PORTABLE = "https://github.com/git-for-windows/git/releases/download/v2.53.0.windows.1/PortableGit-2.53.0-64-bit.7z.exe"

Logger = logging.getLogger(__name__)


def download_mingw(path=None, tag="latest", overwrite=False, **_):
    """
    Downloads MinGW64 for Windows
    """
    if platform.system() != "Windows":
        print("MinGW download is only supported on Windows")
        return

    print("Downloading MinGW64")

    output = prepare_output(
        path, env.path("windows", "MinGW64"), overwrite=overwrite, isdir=True
    )
    if not output:
        return

    zipfile = download_file(MINGW, output.parent / "MinGW64.zip")
    avail = unzip(zipfile, path=output.parent, rename=output.name, overwrite=overwrite)

    print(f"Done, now available at: {avail}/bin")
    print(
        "You may need to add it to your PATH environment variable, ISOFIT will also do this automatically at runtime"
    )

    env.changeKey("path.mingw", "{windows}/MinGW64/bin")
    env.save()
    env.load()


def download_portable_git(path=None, overwrite=False, **_):
    """
    Downloads Portable Git for Windows and extracts it
    """
    if platform.system() != "Windows":
        print("Portable Git download is only supported on Windows")
        return

    print("Downloading Portable Git (Windows)")

    output = prepare_output(
        path, env.path("windows", "PortableGit"), overwrite=overwrite, isdir=True
    )
    if not output:
        return

    exe_file = download_file(GIT_PORTABLE, output.parent / "PortableGit.exe")

    try:
        print("Extracting Portable Git (this may take a while)...")
        cmd = f'"{exe_file}" -y -o"{output}"'
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(
            f"Failed to extract Portable Git automatically. Please extract {exe_file} manually to {output}"
        )
        return

    try:
        os.remove(exe_file)
    except Exception:
        pass

    env.changeKey("path.git", "{windows}/PortableGit/bin")
    env.save()
    env.load()

    print(f"Done, now available at: {output}")


def download(path=None, overwrite=False, **_):
    """
    Downloads Windows helper tools (MinGW64 and Portable Git)

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path.
    overwrite : bool, default=False
        Overwrite an existing installation
    **_ : dict
        Ignores unused params for compatibility with download_all()
    """
    download_mingw(path=path, overwrite=overwrite)
    download_portable_git(path=path, overwrite=overwrite)


def validate(path=None, checkForUpdate=True, debug=print, error=print, **_):
    """
    Validates Windows helper tools installation.
    On non-Windows systems, always returns True.

    Parameters
    ----------
    path : str, default=None
        Path to verify (not used for Windows helpers)
    checkForUpdate : bool, default=True
        Checks for updates (not implemented)
    debug : function, default=print
        Print function for debug messages
    error : function, default=print
        Print function for error messages
    **_ : dict
        Ignores unused params for compatibility

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if platform.system() != "Windows":
        debug("[OK] Windows helpers not applicable on non-Windows system")
        return True

    if path is None:
        path = env.windows

    debug(f"Verifying path for Windows Utilities: {path}")

    path = Path(path)

    if not path.exists():
        error("[x] Windows Utilities path does not exist")
        return False

    if not (path / "MinGW64").exists():
        error("[x] MinGW64 not found in Windows Utilities path")
        return False

    if not (path / "PortableGit").exists():
        error("[x] Portable Git not found in Windows Utilities path")
        return False

    debug("[OK] Windows Utilities are properly installed")
    return True


def update(check=False, **kwargs):
    """
    Checks if an update is needed and downloads if required.

    Parameters
    ----------
    check : bool, default=False
        Just check if update is needed, do not download
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
@shared.path(help="Root directory to download Windows helpers to, ie. [path]/windows")
@shared.overwrite
@shared.check
@click.option(
    "--mingw",
    is_flag=True,
    help="Downloads the MinGW64 (for Windows)",
)
@click.option(
    "--git",
    is_flag=True,
    help="Downloads Portable Git (Windows)",
)
def download_cli(mingw, git, **kwargs):
    """\
    Downloads Windows helper tools (MinGW64, Portable Git).
    If no specific tool is selected, downloads both.
    """
    if mingw:
        download_mingw(**kwargs)
    elif git:
        download_portable_git(**kwargs)
    else:
        download(**kwargs)
