"""
Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data
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
    print(f"Downloading ISOFIT data")

    output = prepare_output(output, env.data)
    if not output:
        return

    metadata = release_metadata("isofit", "isofit-data", tag)

    print(f"Pulling release {metadata['tag_name']}")
    zipfile = download_file(metadata["zipball_url"], output.parent / "isofit-data.zip")

    print(f"Unzipping {zipfile}")
    avail = unzip(zipfile, path=output.parent, rename=output.name)

    print(f"Done, now available at: {avail}")


@cli.download.command(name="data")
@cli.output(help="Root directory to download data files to, ie. [path]/data")
@cli.tag
def download_cli(**kwargs):
    """\
    Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data.

    \b
    Run `isofit download paths` to see default path locations.
    There are two ways to specify output directory:
        - `isofit --data /path/data download data`: Override the ini file. This will save the provided path for future reference.
        - `isofit download data --output /path/data`: Temporarily set the output location. This will not be saved in the ini and may need to be manually set.
    It is recommended to use the first style so the download path is remembered in the future.
    """
    download(**kwargs)


def validate(path=None, **_):
    """
    Validates an ISOFIT data installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path
    **_ : dict
        Ignores unused params that may be used by other validate functions. This is to
        maintain compatibility with env.validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if path is None:
        path = env.data

    print(f"Verifying path for ISOFIT data: {path}")

    if not (path := Path(path)).exists():
        print(
            "Error: Path does not exist, please download it via `isofit download data`"
        )
        return False

    # Just validate some key files
    check = [
        "earth_sun_distance.txt",
        "emit_model_discrepancy.mat",
        "testrfl.dat",
    ]
    files = list(path.glob("*"))
    if not all([path / file in files for file in check]):
        print(
            "Error: ISOFIT data do not appear to be installed correctly, please ensure it is"
        )
        return False

    print("Path is valid")
    return True


@cli.validate.command(name="data")
@cli.path(help="Path to an ISOFIT data installation")
def validate_cli(**kwargs):
    """\
    Validates an ISOFIT data installation
    """
    validate(**kwargs)
