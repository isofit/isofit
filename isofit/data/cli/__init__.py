import click

from isofit.data.download import cli_download

from .data import download as data
from .examples import download as examples
from .sixs import download as sixs
from .srtmnet import download as srtmnet


@cli_download.command(name="all")
def download_all():
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the isofit.ini file using latest tags and versions.
    """
    funcs = [data, examples, sixs, srtmnet]
    pad = "=" * 16

    for i, func in enumerate(funcs):
        click.echo(f"{pad} Beginning download {i+1} of {len(funcs)} {pad}")
        func()
        click.echo()

    click.echo("Finished all downloads")
