import click

from .ini import Ini

env = Ini()

# Attach download commands
import isofit.data.cli


@click.command(name="path")
@click.argument("product", type=click.Choice(env._dirs))
@click.option("-k", "--key", help="Append the product's key")
def cli(product, key=None):
    """\
    Prints the path to a specific product
    """
    path = env[product]
    if key:
        # Check if the given key exists
        if not (sub := env[key]):
            # If it doesn't, check if this key is a subkey of the product
            # eg. product=srtmnet, key=file, actual=srtmnet.file
            subkey = f"{product}.{key}"
            if not (sub := env[subkey]):
                print(
                    f"The key {key!r} does not exist in the INI nor does the subkey {subkey!r}"
                )
                return

        path += f"/{sub}"

    print(path)
