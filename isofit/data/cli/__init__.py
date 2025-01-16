import importlib
import pkgutil

from isofit.data import env
from isofit.data.download import cli

# Auto-discovers the submodules of isofit.data.cli
Modules = {
    name: importlib.import_module(f".{name}", __spec__.name)
    for imp, name, _ in pkgutil.iter_modules(__path__)
}

# Remove disabled modules
Modules = {
    name: module
    for name, module in Modules.items()
    if not getattr(module, "DISABLED", False)
}


def runOnAll(func, **kwargs):
    """
    Executes a function of each loaded module

    Parameters
    ----------
    func : str
        Name of the function to invoke from the module
    **kwargs : dict
        Key-word arguments to pass to the function
    """
    pad = "=" * 16

    for i, module in enumerate(Modules.values()):
        print(f"{pad} Beginning {func} {i+1} of {len(Modules)} {pad}")

        getattr(module, func)(**kwargs)

        print()

    print("Finished all processes")


@cli.download.command(name="all")
@cli.check
@cli.overwrite
def download_all(**kwargs):
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the
    isofit.ini file using latest tags and versions
    """
    if kwargs.get("overwrite"):
        runOnAll("download", **kwargs)
    else:
        runOnAll("update", **kwargs)


@cli.validate.command(name="all")
def validate_all(**kwargs):
    """\
    Validates all ISOFIT extra dependencies at the locations specified in the
    isofit.ini file as well as check for updates using latest tags and versions
    """
    runOnAll("validate", **kwargs)


def env_validate(keys, **kwargs):
    """
    Utility function for the `env` object to quickly validate specific dependencies

    Parameters
    ----------
    keys : list
        List of validator functions to call
    """
    error = kwargs.get("error", print)

    # Turn off checking for updates when using this function by default
    # This makes env.path less verbose
    kwargs["checkForUpdate"] = kwargs.get("checkForUpdate", False)

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
