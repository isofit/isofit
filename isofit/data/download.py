"""
Utility functions for the downloader modules
"""

import io
import json
import os
import shutil
import tarfile
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from zipfile import ZipFile

import click

from isofit.data import env


def release_metadata(org, repo, tag="latest"):
    """
    Fetch GitHub metadata for the latest tagged release.

    Credit to Kevin Wurster https://github.com/isofit/isofit/pull/448#issuecomment-1966747551

    Parameters
    ----------
    org : str
        GitHub organization name
    repo : str
        GitHub repository name
    tag : str
        Release tag to pull

    Returns
    -------
    dict
        Metadata returned by the retrieved release
    """
    url = f"https://api.github.com/repos/{org}/{repo}/releases/{tag}"
    response = urllib.request.urlopen(url)

    encoding = response.headers.get_content_charset()
    payload = response.read()

    return json.loads(payload.decode(encoding))


def download_file(url, dstname=None, overwrite=True):
    """
    Stream downloads a file

    Parameters
    ----------
    url : str
        URL to download
    dstname : str
        Destination file name
    overwrite : bool, default=True
        Overwrite the destination file if it already exists

    Returns
    -------
    outfile : str
        Output downloaded filepath
    """
    response = urllib.request.urlopen(url)

    total = 0
    if length := response.info()["Content-Length"]:
        total = int(length)

    # Using Python's 'email' module for this is certainly odd, but due to an
    # upcoming deprecation, this is actually the officially recommended way
    # to do this: https://docs.python.org/3/library/cgi.html#cgi.parse_header
    msg = EmailMessage()
    msg["Content-Disposition"] = response.headers["Content-Disposition"]

    outfile = Path(dstname or msg["Content-Disposition"].params["filename"])
    if outfile.exists() and not overwrite:
        raise FileExistsError(outfile)

    with click.progressbar(length=total, label="Downloading file") as bar:
        with open(outfile, "wb") as file:
            while chunk := response.read(io.DEFAULT_BUFFER_SIZE):
                file.write(chunk)
                bar.update(io.DEFAULT_BUFFER_SIZE)

    return outfile


def unzip(file, path=None, rename=None, overwrite=False, cleanup=True):
    """
    Unzips a zipfile

    Parameters
    ----------
    path : str, default=None
        Path to extract the zipfile to. Defaults to the directory the zip is found in
    rename : str, default=None
        Renames the extracted data to this
    overwrite : bool, default=False
        Overwrites the path destination with the zip contents if enabled
    cleanup : bool, default=True
        Removes the zip file after completion

    Returns
    -------
    dst : str
        The extracted output path
    """
    path = Path(path or os.path.dirname(file))

    with ZipFile(file) as z:
        name = z.namelist()[0]

        # Verify the output target doesn't exist
        dst = path / (rename or name)
        if dst.exists() and not overwrite:
            raise FileExistsError(dst)

        z.extractall(path)

    src = Path(path) / name
    if rename:
        if dst.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
        else:
            shutil.move(src, dst)

    if cleanup:
        os.remove(file)

    return dst


def untar(file, output):
    """
    Untars a .tar file. Removes the tar file after extracting

    Parameters
    ----------
    file : str
        .tar file to extract
    output : str
        Path to output to

    Returns
    -------
    output : str
        The extracted output path
    """
    with tarfile.open(file) as tar:
        tar.extractall(path=output)

    os.remove(file)

    return output


def prepare_output(output, default, isdir=False, overwrite=False):
    """
    Prepares the output path by ensuring the parents exist and itself doesn't presently exist.

    Parameters
    ----------
    output : str | None
        Path to download to
    default : str
        Default path defined by the ini file
    isdir : bool, default=False
        This is supposed to be a directory
    overwrite : bool, default=False
        Ignore if the output already exists
    """
    if not output:
        output = default

    output = Path(output)

    print(f"Output as: {output}")

    if not overwrite and output.exists():
        print(
            f"Path already exists, please remove it or set the overwrite flag if you would like to redownload"
        )
        return

    try:
        if isdir:
            output.mkdir(parents=True, exist_ok=True)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory: {e}")
        return

    return output


@click.group("download", invoke_without_command=True, no_args_is_help=True)
def cli():
    """\
    Download extra ISOFIT files that do not come with the default installation
    """
    pass


@cli.command(name="paths")
def preview_paths():
    """\
    Preview download path locations. Paths can be changed from the default by using the overrides on the `isofit` command. See more via `isofit --help`

    \b
    Example:
    Change the default `data` and `examples` paths
    $ isofit --data /path/to/data --examples /different/path/examples download paths
        Download paths will default to:
        - data = /path/to/data
        - examples = /different/path/examples
    \b
    These will be saved and may be reviewed:
    $ isofit download paths
        Download paths will default to:
        - data = /path/to/data
        - examples = /different/path/examples
    """
    print("Download paths will default to:")
    for key, path in env.items("dirs"):
        print(f"- {key} = {path}")
