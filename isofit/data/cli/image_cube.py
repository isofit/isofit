"""
Downloads the extra ISOFIT image cube data files from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/
"""

from pathlib import Path

import click

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output, unzip

CMD = "imagecube"
URL = "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/{size}_chunk.zip"


def download(path=None, size="both", overwrite=False, **_):
    """
    Downloads the extra ISOFIT data files from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/.

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path.
    size : "both" | "small" | "medium"
        Which chunk size to pull
    overwrite : bool, default=False
        Overwrite an existing installation
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions
    """
    if size == "both":
        download(path, "small", overwrite)
        download(path, "medium", overwrite)
        return

    if size not in ("small", "medium"):
        raise AttributeError(
            f"Image cube chunk size must be either 'small' or 'medium', got: {size}"
        )

    print(f"Downloading ISOFIT image cube data: {size}")

    output = Path(path or env.imagecube) / size
    output = prepare_output(output, None, overwrite=overwrite)
    if not output:
        return

    url = URL.format(size=size)

    print(f"Pulling {url}")
    zipfile = download_file(url, output.parent / f"{size}_chunk.zip")

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name, overwrite=overwrite)

    print(f"Done, now available at: {avail}")


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

    Notes
    -----
    The Github workflows watch for the string "[x]" to determine if the cache needs to
    update the data of this module. If your module does not include this string, the
    workflows will never detect updates.
    """
    if size == "both":
        return validate(path, "small") & validate(path, "medium")

    if path is None:
        path = env.imagecube
    path = Path(path)

    debug(f"Verifying path for ISOFIT {size} image cube: {path}")

    sizes = {"small": "7000-7010", "medium": "7k-8k"}
    for kind in ("loc", "obs", "rdn"):
        file = path / size / f"ang20170323t202244_{kind}_{sizes[size]}"
        if not file.exists():
            error(
                f"[x] ISOFIT {size} image cube data do not appear to be installed correctly"
            )
            error(f"[x] Missing file: {file}")
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


# Shared click options
size = click.option(
    "-s",
    "--size",
    type=click.Choice(["small", "medium", "both"]),
    default="both",
    help="Chunk size",
)


@shared.download.command(name=CMD)
@shared.path(
    help="Root directory to download image cube data files to, ie. [path]/imagecube"
)
@shared.overwrite
@shared.check
@size
def download_cli(**kwargs):
    """\
    Downloads the extra ISOFIT image cube data files from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --imagecube /path/imagecube download imagecube`: Override the ini file. This will save the provided path for future reference.
        - `isofit download imagecube --path /path/imagecube`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(
    help="Root directory to download image cube data files to, ie. [path]/imagecube"
)
@size
def validate_cli(**kwargs):
    """\
    Validates the installation of the image cube data
    """
    validate(**kwargs)
