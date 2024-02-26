"""
Implements the `isofit download` subcommands
"""

import click

import isofit


@click.group(invoke_without_command=True)
def download():
    """\
    Download extra ISOFIT files that do not come with the default installation
    """
    pass


@download.command(name="data")
@click.option(
    "-o",
    "--output",
    default=f"{isofit.root}/data",
    help="Directory to download ISOFIT data files to",
)
def data(output):
    """\
    Downloads the extra ISOFIT data files from the repository [].
    """
    click.echo(f"Downloading ISOFIT data to: {output}")

    ...

    click.echo("Done")


@download.command(name="examples")
@click.option(
    "-o",
    "--output",
    default=f"{isofit.root}/examples",
    help="Directory to download ISOFIT examples to",
)
def examples(output):
    """\
    Downloads the ISOFIT examples from the repository [].
    """
    click.echo(f"Downloading ISOFIT examples to: {output}")

    ...

    click.echo("Done")


if __name__ == "__main__":
    download()
else:
    from isofit import cli

    cli.add_command(download)
