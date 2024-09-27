"""
Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials
"""

import click

from isofit.data import env
from isofit.data.download import (
    cli_download,
    download_file,
    output,
    prepare_output,
    release_metadata,
    tag,
    unzip,
)


def download(output=None, tag="latest"):
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

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --examples /path/examples download examples`: Override the ini file. This will save the provided path for future reference.
        - `isofit download examples --output /path/examples`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download(**kwargs)
