import pkgutil

from isofit.data import env
from isofit.data.download import cli


def findModules():
    """
    Finds all of the available download modules

    Returns
    -------
    dict
        Dict of valid modules
    """
    return {
        name: imp.find_module(name).load_module(name)
        for imp, name, _ in pkgutil.iter_modules(__path__)
    }


@cli.download.command(name="all")
def download_all():
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the isofit.ini file using latest tags and versions.
    """
    mods = findModules().values()
    pad = "=" * 16

    for i, module in enumerate(mods):
        print(f"{pad} Beginning download {i+1} of {len(mods)} {pad}")
        module.download()
        print()

    print("Finished all downloads")


@cli.validate.command(name="all")
def validate_all():
    """\
    Validates all ISOFIT extra dependencies at the locations specified in the isofit.ini file.
    """
    mods = findModules().values()
    pad = "=" * 16

    for i, module in enumerate(mods):
        print(f"{pad} Validating {i+1} of {len(mods)} {pad}")
        module.validate()
        print()

    print("Finished all validations")


def env_validate(keys, **kwargs):
    """
    Utility function for the `env` object to quickly validate specific dependencies
    """
    mods = findModules()
    all_valid = True
    for key in keys:
        module = mods.get(key)
        if module is None:
            print(f"Product not found: {key}")
            all_valid = False
        else:
            all_valid &= module.validate(**kwargs)

    return all_valid


env.validate = env_validate
