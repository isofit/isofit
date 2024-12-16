"""
Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials
"""

from pathlib import Path

from isofit.data import env
from isofit.data.download import (
    cli,
    download_file,
    prepare_output,
    release_metadata,
    unzip,
)

NEON_URL = "https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/tutorials/subset_data.zip"


def download_neon(examples):
    """
    Downloads the NEON dataset from https://avng.jpl.nasa.gov/pub/PBrodrick/isofit/tutorials/subset_data.zip.

    Parameters
    ----------
    examples : Path
        Path to the examples directory
    """
    print("Downloading NEON data for the example")

    output = prepare_output(examples / "isotuts/NEON/data", ...)
    if not output:
        return

    zipfile = download_file(NEON_URL, output.parent / "NEON-subset-data.zip")

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name)


def download(output=None, tag="latest", overwrite=False):
    """
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.

    Parameters
    ----------
    output : str | None
        Path to output as. If None, defaults to the ini path.
    tag : str
        Release tag to pull from the github.
    overwrite : bool, default=False
        Overwrite an existing installation
    """
    print(f"Downloading ISOFIT examples")

    output = prepare_output(output, env.examples)
    if not output:
        return

    metadata = release_metadata("isofit", "isofit-tutorials", tag)

    print(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(
        metadata["zipball_url"], output.parent / "isofit-tutorials.zip", overwrite=overwrite
    )

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name)

    env.changeVersion("examples", metadata["tag_name"])

    download_neon(output)

    print(f"Done, now available at: {avail}")


@cli.download.command(name="examples")
@cli.output(help="Root directory to download ISOFIT examples to, ie. [path]/examples")
@cli.tag
@cli.overwrite
def download_cli(**kwargs):
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


def validate(path=None, debug=print, error=print, **_):
    """
    Validates an ISOFIT examples installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path
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
        error(
            "Error: Examples path does not exist, please download it via `isofit download examples`"
        )
        return False

    expected = [
        "20151026_SantaMonica",
        "20171108_Pasadena",
        "20190806_ThermalIR",
        "LICENSE",
        "README.md",
        "image_cube",
        "profiling_cube",
        "py-hypertrace",
    ]
    if not list(path.glob("*")) != expected:
        error(
            "Error: ISOFIT examples do not appear to be installed correctly, please ensure it is"
        )
        return False

    debug("Path is valid")

    checkForUpdate(error)

    return True


def checkForUpdate(path, tag="latest", print=print):
    """
    Checks the installed version against the latest release

    Parameters
    ----------
    path : str, default=None
        Path to update. If None, defaults to the ini path
    print : function, default=print
        Print function to use for messages

    Returns
    -------
    bool
        True if there is a version update, else False
    """
    file = Path(path / "version.txt")
    if not file.exists():
        print("Failed to find a version.txt file under the given path. Version is unknown. It is recommended to redownload via `isofit download examples --overwrite`")
        return True

    version = env.versions.get("examples")
    metadata = release_metadata("isofit", "isofit-tutorials", tag)

    if version != (latest := metadata["tag_name"]):
        print(f"Your examples are out of date and may cause issues. Latest is {latest}, currently installed is {version}. Please update via `isofit download updates`")
        return True

    return False



def update(path=None, tag="latest"):
    """
    TODO

    Parameters
    ----------
    path : str, default=None
        Path to update. If None, defaults to the ini path.

    Returns
    -------

    """
    if checkForUpdate()
    download(output=path, tag=tag, overwrite=True)


@cli.validate.command(name="examples")
@cli.path(help="Path to an ISOFIT examples installation")
def validate_cli(**kwargs):
    """\
    Validates an ISOFIT examples installation
    """
    validate(**kwargs)
