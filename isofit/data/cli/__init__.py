import importlib
import pkgutil

from isofit.data import env, shared

# Auto-discovers the submodules of isofit.data.shared
Modules = {
    name: importlib.import_module(f".{name}", __spec__.name)
    for imp, name, _ in pkgutil.iter_modules(__path__)
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


@shared.download.command(name="all")
@shared.check
@shared.overwrite
def download_all(**kwargs):
    """\
    Downloads all ISOFIT extra dependencies to the locations specified in the
    isofit.ini file using latest tags and versions
    """
    if kwargs.get("overwrite"):
        runOnAll("download", **kwargs)
    else:
        runOnAll("update", **kwargs)


@shared.validate.command(name="all")
def validate_all(**kwargs):
    """\
    Validates all ISOFIT extra dependencies at the locations specified in the
    isofit.ini file as well as check for updates using latest tags and versions
    """
    runOnAll("validate", **kwargs)


def env_validate(keys, quiet=False, **kwargs):
    """
    Utility function for the `env` object to quickly validate specific dependencies

    Parameters
    ----------
    keys : list
        List of validator functions to call
    quiet : bool, default=False
        Silences the error and debug messages of the validation functions

    Examples
    --------
    >>> env.validate("all")
    >>> env.validate("isoplots", path=env.plots, quiet=True)
    >>> env.validate(["data", "examples"])
    >>> env.validate(["data", "examples"], error=Logger.error, debug=Logger.debug)
    """
    error = kwargs.get("error", print)

    # Turn off checking for updates when using this function by default
    # This makes env.path less verbose
    kwargs["checkForUpdate"] = kwargs.get("checkForUpdate", False)

    if isinstance(keys, str):
        if keys == "all":
            keys = Modules
        else:
            keys = [keys]

    if quiet:
        kwargs["error"] = lambda _: ...
        kwargs["debug"] = lambda _: ...

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
