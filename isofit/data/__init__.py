import click

from .ini import Ini

env = Ini()

# Attach download commands
import isofit.data.cli


@click.command(name="path")
@click.argument("product", type=click.Choice(env.dirs))
def cli_env_path(product):
    """\
    Prints the path to a specific product
    """
    print(env[product])
