"""
Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials
"""

from pathlib import Path

from isofit.data import env, shared
from isofit.data.download import download_file, prepare_output, release_metadata, unzip

CMD = "examples"
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

    output = prepare_output(examples / "isotuts/NEON/data", "./neon_data")
    if not output:
        return

    zipfile = download_file(NEON_URL, output.parent / "NEON-subset-data.zip")

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name)


def download(path=None, tag="latest", overwrite=False, **_):
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
    print(f"Downloading ISOFIT examples")

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

    download_neon(output)

    print(f"Done, now available at: {avail}")
    print("[!] Be sure to build the examples for your system via `isofit build`")


def validate(path=None, checkForUpdate=True, debug=print, error=print, **_):
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
    if not validate(**kwargs):
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
def download_cli(**kwargs):
    """\
    Downloads the ISOFIT examples from the repository https://github.com/isofit/isofit-tutorials.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --examples /path/examples download examples`: Override the ini file. This will save the provided path for future reference.
        - `isofit download examples --path /path/examples`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    if kwargs.get("overwrite"):
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
    validate(**kwargs)
