"""
Downloads the extra ISOFIT image cube data files from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/
"""

from pathlib import Path

import click

from isofit.data import env
from isofit.data.download import cli, download_file, prepare_output, unzip

URL = "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/{size}_chunk.zip"
Chunk = click.option(
    "-s",
    "--size",
    type=click.Choice(["small", "medium", "both"]),
    default="both",
    help="Chunk size",
)


def download(output=None, size="both"):
    """
    Downloads the extra ISOFIT data files from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/.

    Parameters
    ----------
    output : str | None
        Path to output as. If None, defaults to the ini path.
    size : "both" | "small" | "medium"
        Which chunk size to pull
    """
    if size == "both":
        download(output, "small")
        download(output, "medium")
        return

    if size not in ("small", "medium"):
        raise AttributeError(
            f"Image cube chunk size must be either 'small' or 'medium', got: {size}"
        )

    print(f"Downloading ISOFIT image cube data: {size}")

    output = Path(output or env.imagecube) / size
    output = prepare_output(output, None)
    if not output:
        return

    url = URL.format(size=size)

    print(f"Pulling {url}")
    zipfile = download_file(url, output.parent / f"{size}_chunk.zip")

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name)

    print(f"Done, now available at: {avail}")


@cli.download.command(name="imagecube")
@cli.output(
    help="Root directory to download image cube data files to, ie. [path]/imagecube"
)
@Chunk
def download_cli(**kwargs):
    """\
    Downloads the extra ISOFIT image cube data files from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --imagecube /path/imagecube download imagecube`: Override the ini file. This will save the provided path for future reference.
        - `isofit download imagecube --output /path/imagecube`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download(**kwargs)


def validate(path=None, size="both", debug=print, error=print, **_):
    """
    Validates an ISOFIT image cube data installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path
    size : "both" | "small" | "medium"
        Which chunk size to validate
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
    if size == "both":
        return validate(path, "small") & validate(path, "medium")

    if path is None:
        path = Path(env.imagecube)

    debug(f"Verifying path for ISOFIT {size} image cube: {path}")

    sizes = {"small": "7000-7010", "medium": "7k-8k"}
    for kind in ("loc", "obs", "rdn"):
        file = path / size / f"ang20170323t202244_{kind}_{sizes[size]}"
        if not file.exists():
            error(
                f"Error: ISOFIT {size} image cube data do not appear to be installed correctly, please ensure it is"
            )
            error(f"Missing file: {file}")
            return False

    debug("Path is valid")
    return True


@cli.validate.command(name="imagecube")
@cli.path(help="Path to an image cube installation")
@Chunk
def validate_cli(**kwargs):
    """\
    Validates an ISOFIT image cube data installation
    """
    validate(**kwargs)
