"""
Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials
"""

from pathlib import Path

import click

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output, release_metadata, unzip

ESSENTIAL = False
CMD = "examples"

# url: path to download
Extras = {
    "neon": {
        "url": "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/tutorials/subset_data.zip",
        "subpath": "NEON/data",
    },
    "image_cube": {
        "url": "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/{size}_chunk.zip",
        "subpath": "image_cube/{size}",
    },
    "lakemary": {
        "url": "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/LakeMary.zip",
        "subpath": "LakeMary",
    },
}


def download_extra(name, path=None, overwrite=False, **kwargs):
    """
    Downloads additional data for the examples. Assumes the data is a zip.

    Parameters
    ----------
    path : Path
        Path to the examples directory
    size : str
        Specify which of the two image_cube datasets to download.
    overwrite : bool
        Flag to overwrite current installed files
    **kwargs : dict
        Additional key-word arguments used by one or more extras. For instance,
        image_cube uses ``size`` to format the url and output subpaths
    """
    if name not in Extras:
        raise AttributeError(f"Extra {name!r} is not one of: {list(Extras)}")

    extra = Extras[name]

    # ImageCube is a special case that downloads two kinds
    if name == "image_cube":
        size = kwargs.get("size")
        if size == "both":
            download_extra("image_cube", path, overwrite, size="small")
            download_extra("image_cube", path, overwrite, size="medium")
            return

        if size not in ("small", "medium"):
            raise AttributeError(
                f"Image cube chunk size must be either 'small' or 'medium', got: {size!r}"
            )

    output = Path(path or env.examples) / extra["subpath"].format(**kwargs)
    output = prepare_output(output, None, overwrite=overwrite)
    if not output:
        return

    url = extra["url"].format(**kwargs)
    zipfile = download_file(url, output.parent / f"{output.name}.zip")

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name, overwrite=overwrite)

    print(f"Done, now available at: {avail}")


def download_tutorials(path=None, tag="latest", overwrite=False, **_):
    """
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path
    tag : str
        Release tag to pull from the github
    overwrite : bool, default=False
        Overwrite an existing installation
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions
    """
    output = prepare_output(path, env.examples, overwrite=overwrite)
    if not output:
        return

    metadata = release_metadata("isofit", "isofit-tutorials", tag)

    print(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(
        metadata["zipball_url"], output.parent / "isofit-tutorials.zip"
    )

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name, overwrite=overwrite)

    with open(output / "version.txt", "w") as file:
        file.write(metadata["tag_name"])


def download(path=None, overwrite=False, **_):
    """
    Downloads all examples

    Parameters
    ----------
    path : str | None
        Path to output as. If None, defaults to the ini path
    overwrite : bool, default=False
        Overwrite an existing installation
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions
    """
    print(f"Downloading ISOFIT examples")
    download_tutorials(path=path, overwrite=overwrite)
    download_extra("neon", path=path, overwrite=overwrite)
    download_extra("image_cube", path=path, overwrite=overwrite, size="both")
    download_extra("lakemary", path=path, overwrite=overwrite)

    print("[!] Be sure to build the examples for your system via `isofit build`")


def validate_tutorials(path=None, checkForUpdate=True, debug=print, error=print, **_):
    """
    Validates an ISOFIT examples installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path
    checkForUpdate : bool, default=True
        Checks for updates if the path is valid
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
    if path is None:
        path = env.examples

    debug(f"Verifying path for ISOFIT examples: {path}")

    if not (path := Path(path)).exists():
        error("[x] Examples path does not exist")
        return False

    expected = set(
        [
            "20151026_SantaMonica",
            "20171108_Pasadena",
            "20190806_ThermalIR",
            "LICENSE",
            "README.md",
            "image_cube",
            "profiling_cube",
        ]
    )
    names = set([file.name for file in path.glob("*")])
    if missing := (expected - names):
        error("[x] ISOFIT examples do not appear to be installed correctly")
        debug(f"Expected: {expected}")
        debug(f"Got: {names}")
        debug(f"Missing: {missing}")
        return False

    debug("[OK] Path is valid")

    if checkForUpdate:
        return isUpToDate(path, debug=debug, error=error)

    return True


def validate_image_cube(path=None, size="both", debug=print, error=print, **_):
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
        small = validate_image_cube(path, "small", debug=debug, error=error)
        medium = validate_image_cube(path, "medium", debug=debug, error=error)
        return small & medium

    if path is None:
        path = Path(path or env.examples) / "image_cube" / size

    debug(f"Verifying path for ISOFIT {size} image cube: {path}")

    sizes = {"small": "7000-7010", "medium": "7k-8k"}
    for kind in ("loc", "obs", "rdn"):
        file = path / f"ang20170323t202244_{kind}_{sizes[size]}"
        if not file.exists():
            error(
                f"[x] ISOFIT {size} image cube data do not appear to be installed correctly"
            )
            error(f"[x] Missing file: {file}")
            return False

    debug("[OK] Path is valid")
    return True


def validate(**kwargs):

    return validate_tutorials(**kwargs) and validate_image_cube(**kwargs)


def isUpToDate(path=None, tag="latest", debug=print, error=print, **_):
    """
    Checks the installed version against the latest release

    Parameters
    ----------
    path : str, default=None
        Path to update. If None, defaults to the ini path
    debug : function, default=print
        Print function to use for debug messages, eg. logging.debug
    error : function, default=print
        Print function to use for error messages, eg. logging.error
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with other functions

    Returns
    -------
    bool
        True if the path is up to date, False otherwise

    Notes
    -----
    The Github workflows watch for the string "[x]" to determine if the cache needs to
    update the data of this module. If your module does not include this string, the
    workflows will never detect updates.
    """
    if path is None:
        path = env.examples

    debug(f"Checking for updates for examples on path: {path}")

    file = Path(path) / "version.txt"
    if not file.exists():
        error(
            "[x] Failed to find a version.txt file under the given path. Version is unknown"
        )
        return False

    metadata = release_metadata("isofit", "isofit-tutorials", tag)
    with open(file, "r") as f:
        current = f.read()

    if current != (latest := metadata["tag_name"]):
        error(f"[x] Latest is {latest}, currently installed is {current}")
        return False

    debug(f"[OK] Path is up to date, current version is: {current}")

    return True


def update(check=False, **kwargs):
    """
    Checks for an update and executes a new download if it is needed

    Parameters
    ----------
    check : bool, default=False
        Just check if an update is available, do not download
    **kwargs : dict
        Additional key-word arguments to pass to download()
    """
    debug = kwargs.get("debug", print)
    check = validate_tutorials(**kwargs) and validate_image_cube(**kwargs)
    if not check:
        if not check:
            kwargs["overwrite"] = True
            debug("Executing update")
            download(**kwargs)
        else:
            debug(f"Please download the latest via `isofit download {CMD}`")


@shared.download.command(name=CMD)
@shared.path(help="Root directory to download example files to, ie. [path]/examples")
@shared.tag
@shared.overwrite
@shared.check
@click.option(
    "--tutorials",
    is_flag=True,
    help="Downloads only the tutorials (no datasets) to the examples directory",
)
@click.option(
    "--neon",
    is_flag=True,
    help="Downloads only the NEON dataset to the examples directory",
)
@click.option(
    "--image_cube",
    type=click.Choice(["both", "small", "medium"]),
    flag_value="both",
    default=None,
    help="Downloads only the image_cube dataset to the examples directory",
)
@click.option(
    "--lakemary",
    is_flag=True,
    help="Downloads only the LakeMary dataset to the examples directory",
)
def download_cli(tutorials, neon, image_cube, lakemary, **kwargs):
    """\
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --path examples /path/examples download examples`: Override the ini file. This will save the provided path for future reference.
        - `isofit download examples --path /path/examples`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    dl_all = True
    path = kwargs.get("path") or env.examples
    overwrite = kwargs.get("overwrite", False)

    if tutorials:
        download_tutorials(
            tag=kwargs.get("tag", "latest"), overwrite=kwargs.get("overwrite", False)
        )
        dl_all = False

    if neon:
        download_extra("neon", path=path, overwrite=overwrite)
        dl_all = False

    if image_cube:
        download_extra(
            "image_cube",
            examples=path,
            overwrite=overwrite,
            size=image_cube,
        )
        dl_all = False

    if lakemary:
        download_extra(
            "lakemary",
            path=path,
            overwrite=overwrite,
        )
        dl_all = False

    # Only execute if the --[extra] flags weren't used
    if dl_all:
        if overwrite:
            download(**kwargs)
        else:
            update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Root directory to download example files to, ie. [path]/examples")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of the ISOFIT examples as well as checks for updates
    """
    validate_tutorials(**kwargs)
    validate_image_cube(**kwargs)
