"""
Downloads vSmartMOM using juliacall
"""

from isofit.data import env, shared

ESSENTIAL = False
Version = "v2.2.0"
CMD = "vsmartmom"


def download(path=None, overwrite=False, **_):
    """
    Validates and sets up the vSmartMOM Julia environment for ISOFIT.
    """
    print("Checking for juliacall and vSmartMOM setup...")

    try:
        from juliacall import Main as jl
    except ImportError:
        raise ImportError(
            "juliacall is required for vSmartMOM usage in ISOFIT. "
            "You can install via: pip install isofit[juliacall]"
        )
    jl.seval("using Pkg")
    jl.seval("Pkg.instantiate()")
    jl.seval(f'Pkg.add(name="vSmartMOM", version="{Version}")')

    env.changeKey("vsmartmom-version", Version)
    env.save()


def validate(path=None, debug=print, error=print, **_):
    """
    Validates the vSmartMOM and juliacall installation
    """

    try:
        from juliacall import Main as jl

        jl.seval("using vSmartMOM")
    except ImportError:
        error(
            "[x] vSmartMOM is not installed. Import error for juliacall. You can install via: pip install isofit[juliacall]"
        )
        return False
    except Exception as e:
        error(f"[x] vSmartMOM could not be loaded: {e}")
        return False

    debug("[OK] Path is valid")
    return True


def update(check=False, **kwargs):
    """
    Checks for an update and executes a new download if it is needed
    Note: Not implemented for this module at this time

    Parameters
    ----------
    check : bool, default=False
        Just check if an update is available, do not download
    **kwargs : dict
        Additional key-word arguments to pass to download()
    """
    debug = kwargs.get("debug", print)
    if not validate(**kwargs):
        if not check:
            kwargs["overwrite"] = True
            debug("Executing update")
            download(**kwargs)
        else:
            debug(f"Please download the latest via `isofit download {CMD}`")


@shared.download.command(name=CMD)
@shared.path(help="Path to local vSmartMOM Julia project directory (optional)")
@shared.tag
@shared.overwrite
@shared.check
def download_cli(**kwargs):
    """\
    Sets up vSmartMOM for use in ISOFIT.
    """
    if kwargs.get("overwrite"):
        download(**kwargs)
    else:
        update(**kwargs)


@shared.validate.command(name=CMD)
@shared.path(help="Path to local vSmartMOM Julia project directory (optional)")
@shared.tag
def validate_cli(**kwargs):
    """\
    Validates the installation of vSmartMOM and juliacall
    """
    validate(**kwargs)
