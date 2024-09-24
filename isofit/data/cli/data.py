"""
Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data
"""

import click

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
    download(**kwargs)
