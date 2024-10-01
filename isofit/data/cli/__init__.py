import click

from isofit.data.cli import data, examples, hypertrace, sixs, srtmnet
from isofit.data.download import cli


@cli.download.command(name="all")
def download_all():
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the isofit.ini file using latest tags and versions.
    """
    modules = [data, hypertrace, examples, sixs, srtmnet]
    pad = "=" * 16

    for i, module in enumerate(modules):
        click.echo(f"{pad} Beginning download {i+1} of {len(modules)} {pad}")
        module.download()
        click.echo()

    click.echo("Finished all downloads")


@cli.validate.command(name="all")
def validate_all():
    """\
    Validates all ISOFIT extra dependencies at the locations specified in the isofit.ini file.
    """
    modules = [data, hypertrace, examples, sixs, srtmnet]
    pad = "=" * 16

    for i, module in enumerate(modules):
        click.echo(f"{pad} Validating {i+1} of {len(modules)} {pad}")
        module.validate()
        click.echo()

    click.echo("Finished all validations")
