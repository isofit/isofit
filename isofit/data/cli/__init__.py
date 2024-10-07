import click

from isofit.data import env
from isofit.data.cli import data, examples, sixs, srtmnet
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


def env_validate(keys):
    """
    Utility function for the `env` object to quickly validate specific dependencies
    """
    all_valid = True
    for key in keys:
        module = globals().get(key)
        if module is None:
            print(f"Product not found: {key}")
            all_valid = False

        all_valid &= module.validate()

    return all_valid


env.validate = env_validate
