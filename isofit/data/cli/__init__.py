import importlib
import pkgutil

from isofit.data import env
from isofit.data.download import cli

# Auto-discovers the submodules of isofit.data.cli
Modules = {
    name: importlib.import_module(f".{name}", __spec__.name)
    for imp, name, _ in pkgutil.iter_modules(__path__)
}


@cli.download.command(name="all")
def download_all():
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the isofit.ini file using latest tags and versions.
    """
    pad = "=" * 16

    for i, module in enumerate(Modules.values()):
        print(f"{pad} Beginning download {i+1} of {len(Modules)} {pad}")
        module.download()
        print()

    print("Finished all downloads")


@cli.validate.command(name="all")
def validate_all():
    """\
    Validates all ISOFIT extra dependencies at the locations specified in the isofit.ini file.
    """
    pad = "=" * 16

    for i, module in enumerate(Modules.values()):
        print(f"{pad} Validating {i+1} of {len(Modules)} {pad}")
        module.validate()
        print()

    print("Finished all validations")


def env_validate(keys, **kwargs):
    """
    Utility function for the `env` object to quickly validate specific dependencies

    Parameters
    ----------
    keys : list
        List of validator functions to call
    """
    error = kwargs.get("error", print)

    all_valid = True
    for key in keys:
        module = Modules.get(key)
        if module is None:
            error(f"Product not found: {key}")
            all_valid = False
        else:
            all_valid &= module.validate(**kwargs)

    return all_valid


env.validate = env_validate
