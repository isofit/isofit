"""
Implements the `isofit download` subcommands
"""

import io
import json
import os
import urllib.request
from email.message import EmailMessage
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import click

from isofit.core import env


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


def prepare_output(output, default):
    """
    Prepares the output path by ensuring the parents exist and itself doesn't presently exist.

    Parameters
    ----------
    output: str | None
        Path to download to
    default: str
        Default path defined by the ini file
    """
    if not output:
        output = default

    output = Path(output)

    click.echo(f"Output as: {output}")

    if output.exists():
        click.echo(
            f"Path already exists, please remove it if you would like to redownload"
        )
        return

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        click.echo(f"Failed to create output directory: {e}")
        return

    return output


# Main command


@click.group("download", invoke_without_command=True)
def cli_download():
    """\
    Download extra ISOFIT files that do not come with the default installation
    """
    pass


# Shared click options

output = partial(click.option, "-o", "--output")
tag = click.option(
    "-t", "--tag", default=f"latest", help="Release tag to pull", show_default=True
)


# Subcommands


def download_data(output=None, tag="latest"):
    """
    Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data.

    Parameters
    ----------
    output: str | None
        Path to output as. If None, defaults to the ini path.
    tag: str
        Release tag to pull from the github.
    """
    click.echo(f"Downloading ISOFIT data")

    output = prepare_output(output, env.data)
    if not output:
        return

    metadata = release_metadata("isofit", "isofit-data", tag)

    click.echo(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(metadata["zipball_url"], output.parent / "isofit-data.zip")

    click.echo(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name)

    click.echo(f"Done, now available at: {avail}")


@cli_download.command(name="data")
@output(help="Root directory to download data files to, ie. [path]/data")
@tag
def cli_data(**kwargs):
    """\
    Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data.

    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --data /path/data download data`: Override the ini file. This will save the provided path for future reference.
        - `isofit download data --output /path/data`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download_data(**kwargs)


def download_examples(output=None, tag="latest"):
    """
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.

    Parameters
    ----------
    output: str | None
        Path to output as. If None, defaults to the ini path.
    tag: str
        Release tag to pull from the github.
    """

    click.echo(f"Downloading ISOFIT examples")

    output = prepare_output(output, env.examples)
    if not output:
        return

    metadata = release_metadata("isofit", "isofit-tutorials", tag)

    click.echo(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(
        metadata["zipball_url"], output.parent / "isofit-tutorials.zip"
    )

    click.echo(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name)

    click.echo(f"Done, now available at: {avail}")


@cli_download.command(name="examples")
@output(help="Root directory to download ISOFIT examples to, ie. [path]/examples")
@tag
def cli_examples(**kwargs):
    """\
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.

    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --examples /path/examples download examples`: Override the ini file. This will save the provided path for future reference.
        - `isofit download examples --output /path/examples`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download_examples(**kwargs)


@cli_download.command(name="all")
def download_all():
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the isofit.ini file using latest tags.
    """
    download_data(env.data, tag="latest")
    download_examples(env.examples, tag="latest")


@cli_download.command(name="paths")
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
    click.echo("Download paths will default to:")
    for key, path in env.items():
        click.echo(f"- {key} = {path}")
