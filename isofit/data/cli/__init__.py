import click

from isofit.data.download import cli_download

from .data import download as download_data
from .examples import download as download_examples
from .sixs import download as download_sixs
from .srtmnet import download as download_srtmnet


@cli_download.command(name="all")
def download_all():
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the isofit.ini file using latest tags and versions.
    """
    download_data(env.data)
    download_examples(env.examples)
    download_sixs(env.srtmnet)
    download_srtmnet(env.srtmnet)
