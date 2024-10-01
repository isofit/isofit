"""
Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data
"""

import click

from isofit.data import env
from isofit.data.download import (
    cli_download,
    cli_opts,
    download_file,
    prepare_output,
    untar,
)

URL = "https://github.com/ashiklom/isofit/releases/download/hypertrace-data/hypertrace-data.tar.gz"


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
    click.echo(f"Downloading Hypertrace data")

    output = prepare_output(output, env.hypertrace)
    if not output:
        return

    file = download_file(URL, output.parent / "hypertrace-data.tar.gz")

    print(file)

    output = untar(file, output)

    click.echo(f"Done, now available at: {output}")


@cli_download.command(name="hypertrace")
@cli_opts.output(help="Root directory to download data files to, ie. [path]/hypertrace")
def cli_hypertrace(**kwargs):
    """\
    Downloads the extra ISOFIT hypertrace data files from https://github.com/ashiklom/isofit/releases/download/hypertrace-data/hypertrace-data.tar.gz.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --data /path/data download hypertrace`: Override the ini file. This will save the provided path for future reference.
        - `isofit download hypertrace --output /path/data`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download(**kwargs)
