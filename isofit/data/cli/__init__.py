import click

from isofit.data.download import cli_download

from .data import download as download_data
from .examples import download as download_examples


@cli_download.command(name="all")
def download_all():
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the isofit.ini file using latest tags.
    """
    download_data(env.data, tag="latest")
    download_examples(env.examples, tag="latest")
