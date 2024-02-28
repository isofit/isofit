"""
Implements the `isofit download` subcommands
"""

import io
import json
import os
import urllib.request
from email.message import EmailMessage
from functools import partial
from zipfile import ZipFile

import click

import isofit


def release_metadata(org, repo, tag="latest"):
    """
    Fetch GitHub metadata for the latest tagged release.

    Credit to Kevin Wurster https://github.com/isofit/isofit/pull/448#issuecomment-1966747551

    Parameters
    ----------
    org: str
        GitHub organization name
    repo: str
        GitHub repository name
    tag: str
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
    Download a file.

    Credit to Kevin Wurster https://github.com/isofit/isofit/pull/448#issuecomment-1966747551

    Parameters
    ----------
    url: str
        URL to download
    dstname: str
        Destination file name
    overwrite: bool, default=True
        Overwrite the destination file if it already exists

    Returns
    -------
    outfile: str
        Output downloaded filepath
    """
    response = urllib.request.urlopen(url)

    # Using Python's 'email' module for this is certainly odd, but due to an
    # upcoming deprecation, this is actually the officially recommended way
    # to do this: https://docs.python.org/3/library/cgi.html#cgi.parse_header
    msg = EmailMessage()
    msg["Content-Disposition"] = response.headers["Content-Disposition"]

    outfile = dstname or msg["Content-Disposition"].params["filename"]
    if os.path.exists(outfile) and not overwrite:
        raise FileExistsError(outfile)

    with open(outfile, "wb") as f:
        while chunk := response.read(io.DEFAULT_BUFFER_SIZE):
            f.write(chunk)

    return outfile


def unzip(file, path=None, rename=None, overwrite=False, cleanup=True):
    """
    Unzips a zipfile

    Parameters
    ----------
    path: str, default=None
        Path to extract the zipfile to. Defaults to the directory the zip is found in
    rename: str, default=None
        Renames the extracted data to this
    overwrite: bool, default=False
        Overwrites the path destination with the zip contents if enabled
    cleanup: bool, default=True
        Removes the zip file after completion

    Returns
    -------
    outp: str
        The extracted output path
    """
    path = path or os.path.dirname(file)

    with ZipFile(file) as z:
        name = z.namelist()[0]

        # Verify the output target doesn't exist
        outp = f"{path}/{rename or name}"
        if os.path.exists(outp) and not overwrite:
            raise FileExistsError(outp)

        z.extractall(path)

    if rename:
        os.rename(f"{path}/{name}", outp)

    if cleanup:
        os.remove(file)

    return outp


# Main command


@click.group("download", invoke_without_command=True)
def cli_download():
    """\
    Download extra ISOFIT files that do not come with the default installation
    """
    pass


# Shared click options

output = partial(click.option, "-o", "--output", default=isofit.root, show_default=True)
tag = click.option(
    "-t", "--tag", default=f"latest", help="Release tag to pull", show_default=True
)


# Subcommands


@cli_download.command(name="data")
@output(help="Root directory to download data files to, ie. [path]/data")
@tag
def data(output, tag):
    """\
    Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data.
    """
    click.echo(f"Downloading ISOFIT data")

    metadata = release_metadata("isofit", "isofit-data", tag)

    click.echo(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(metadata["zipball_url"], f"{output}/isofit-data.zip")

    click.echo(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output, rename="data")

    click.echo(f"Done, now available at: {avail}")


@cli_download.command(name="examples")
@output(help="Root directory to download ISOFIT examples to, ie. [path]/examples")
@tag
def examples(output, tag):
    """\
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.
    """
    click.echo(f"Downloading ISOFIT examples")

    metadata = release_metadata("isofit", "isofit-tutorials", tag)

    click.echo(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(metadata["zipball_url"], f"{output}/isofit-tutorials.zip")

    click.echo(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output, rename="examples")

    click.echo(f"Done, now available at: {avail}")
