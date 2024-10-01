"""
Downloads the extra ISOFIT data files from the repository https://github.com/isofit/isofit-data
"""

from pathlib import Path

from isofit.data import env
from isofit.data.download import cli, download_file, prepare_output, untar

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
    print(f"Downloading Hypertrace data")

    output = prepare_output(output, env.hypertrace)
    if not output:
        return

    file = download_file(URL, output.parent / "hypertrace-data.tar.gz")

    print(file)

    output = untar(file, output)

    print(f"Done, now available at: {output}")


@cli.download.command(name="hypertrace")
@cli.output(help="Root directory to download data files to, ie. [path]/hypertrace")
def download_cli(**kwargs):
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


def validate(path=None):
    """
    Validates a hypertrace data installation

    Parameters
    ----------
    path : str, default=None
        Path to verify. If None, defaults to the ini path

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if path is None:
        path = env.hypertrace

    print(f"Verifying path for hypertrace data: {path}")

    if not (path := Path(path)).exists():
        print(
            "Error: Path does not exist, please download it via `isofit download hypertrace`"
        )
        return False

    subdirs = ["noise", "other", "priors", "reflectance wavelengths"]
    if not list((path / f"hypertrace-data").glob("*")) != subdirs:
        print(
            "Error: Hypertrace does not appear to be installed correctly, please ensure it is"
        )
        return False

    print("Path is valid")
    return True


@cli.validate.command(name="hypertrace")
@cli.path(help="Path to hypertrace data installation")
def validate_cli(**kwargs):
    """\
    Validates a hypertrace data installation
    """
    validate(**kwargs)
