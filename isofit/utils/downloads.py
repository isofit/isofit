"""
Implements the `isofit download` subcommands
"""

import io
import os
import zipfile
from functools import partial

import click
import requests
from github import Github

import isofit


def downloadRelease(repo, output, name):
    """
    Downloads a specific tagged release from a given repository

    Parameters
    ----------
    repo: str
        Repository URL
    output: str
        Directory to download to
    name: str
        Name of the directory. The downloaded directory is based off the Github name,
        so this is used to rename it
    """
    git = Github()
    repo = git.get_repo(repo)
    latest = repo.get_latest_release()
    url = latest.zipball_url

    click.echo(f"Downloading {url}")

    req = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(req.content)) as zip:
        temp = zip.infolist()[0].filename
        zip.extractall(output)

    os.rename(f"{output}/{temp}", f"{output}/{name}")
    click.echo(f"Now available at: {output}/{name}")


# Main command


@click.group(invoke_without_command=True)
def download():
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


@download.command(name="data")
@output(help="Root directory to download data files to, ie. [path]/data")
# @tag
def data(output):
    """\
    Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data.
    """
    click.echo(f"Downloading ISOFIT data")

    downloadRelease("isofit/isofit-data", output, "data")

    click.echo("Done")


@download.command(name="examples")
@output(help="Root directory to download ISOFIT examples to, ie. [path]/examples")
# @tag
def examples(output):
    """\
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.
    """
    click.echo(f"Downloading ISOFIT examples")

    downloadRelease("isofit/isofit-tutorials", output, "examples")

    click.echo("Done")


if __name__ == "__main__":
    download()
else:
    from isofit import cli

    cli.add_command(download)
