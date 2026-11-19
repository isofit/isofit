import logging
import os
import shutil
import time
from datetime import timedelta
from functools import wraps
from os.path import exists
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
from spectral.io import envi as spectral_envi

from isofit.core.common import envi_header

Logger = logging.getLogger(__name__)


def format_args(args: List[str]) -> str:
    """
    Format a list of arguments into a copy-pastable command for the terminal.

    This function takes a list of command-line arguments and formats them into
    a multi-line string with proper backslash continuation, making it easy to
    copy and paste into a terminal. Options and their values are kept on the
    same line for readability.

    Parameters
    ----------
    args : List[str]
        List of command-line arguments to format. The first element should be
        the command name, followed by any flags and arguments.

    Returns
    -------
    str
        A formatted multi-line command string with backslash continuations,
        suitable for terminal execution.

    Examples
    --------
    >>> print(format_args(["python", "script.py", "--input", "file.txt", "--verbose"]))
    python script.py \\
      --input file.txt \\
      --verbose
    """
    lines = [" ".join(args[:2])]

    i = 2
    while i < len(args):
        line = args[i]
        i += 1
        if line.startswith("-") and i < len(args) and not args[i].startswith("-"):
            line += f" {args[i]}"
            i += 1
        lines.append(f"  {line}")

    return " \\\n".join(lines)


F = TypeVar("F", bound=Callable[..., Any])


def log_elapse(func: F) -> F:
    """
    Decorator that logs the elapsed execution time of a function.

    This decorator wraps a function to log debug messages when it starts and
    completes, including the total elapsed time formatted as a timedelta. The
    original function's metadata (name, docstring, etc.) is preserved using
    functools.wraps.

    Parameters
    ----------
    func : callable
        The function to be wrapped and timed.

    Returns
    -------
    callable
        The wrapped function that logs its execution time while preserving
        the original function's signature and return value.

    Examples
    --------
    >>> @log_elapse
    ... def process_data(n):
    ...     time.sleep(1)
    ...     return n * 2
    >>> result = process_data(5)
    DEBUG: Beginning process_data
    DEBUG: Finished process_data in 0:00:01.001234
    >>> result
    10

    Notes
    -----
    The decorator uses ``time.perf_counter()`` for high-resolution timing
    and logs at the DEBUG level, so timing information will only appear
    when debug logging is enabled.
    """

    @wraps(func)  # Preserves the original function's metadata
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        Logger.debug(f"Beginning {func.__name__}")

        beg = time.perf_counter()
        ret = func(*args, **kwargs)
        end = time.perf_counter()

        elapse = end - beg
        Logger.debug(f"Finished {func.__name__} in {timedelta(seconds=elapse)}")

        return ret

    return wrapper  # type: ignore[return-value]


def skip(file: str, envi: bool = True) -> bool:
    """
    Check whether a stage output already exists, logging when it does.

    Parameters
    ----------
    file : str
        Output file to check.
    envi : bool, default=True
        The output is an ENVI file, so its header must exist too.

    Returns
    -------
    bool
        True if the output exists and the work producing it can be skipped.
    """
    done = exists(file) and (not envi or exists(envi_header(file)))
    if done:
        Logger.info(f"Output already exists, skipping: {file}")

    return done


def walk_paths(model) -> list[Path]:
    """
    Every path set in a config, depth-first.

    Parameters
    ----------
    model : pydantic.BaseModel
        The config, or a section of it.

    Returns
    -------
    list[pathlib.Path]
        The set path fields of ``model`` and its subsections.
    """
    found = []

    def walk(value):
        if isinstance(value, Path):
            found.append(value)
        elif hasattr(value, "__pydantic_fields__"):
            # Iterating a model yields its declared fields followed by its extras
            for _, val in value:
                walk(val)
        elif isinstance(value, dict):
            for val in value.values():
                walk(val)
        elif isinstance(value, (list, tuple)):
            for val in value:
                walk(val)

    walk(model)

    return found


def envi_shape(file: str) -> tuple[int, int]:
    """
    Spatial (rows, columns) shape of an ENVI file.
    """
    ds = spectral_envi.open(envi_header(file), file)
    return ds.shape[0], ds.shape[1]


def remove(*files: str) -> None:
    """
    Remove files, ignoring any that do not exist.
    """
    for file in files:
        if exists(file):
            Logger.debug(f"Removing {file}")
            os.remove(file)


def stage(src: Optional[Path], dst: Optional[Path], envi: bool = False) -> None:
    """
    Copy a file into the working directory, skipping it if already staged.

    Parameters
    ----------
    src : pathlib.Path, optional
        File to copy. Nothing is staged when None.
    dst : pathlib.Path, optional
        Where to copy ``src`` to.
    envi : bool, default=False
        ``src`` is an ENVI file, so its header is copied alongside it.

    Raises
    ------
    ValueError
        If ``src`` is set without a ``dst``.
    """
    if src is None:
        return

    if dst is None:
        raise ValueError(f"No destination is configured to stage {src} to")

    if Path(src).resolve() == Path(dst).resolve() or skip(dst, envi=envi):
        return

    Logger.info(f"Staging {src} to {dst}")
    shutil.copyfile(src, dst)
    if envi:
        shutil.copyfile(envi_header(str(src)), envi_header(str(dst)))


def read_wavelengths(
    radiance: Path, file: Optional[Path] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the wavelength centers and FWHM of a radiance cube, in microns.

    Parameters
    ----------
    radiance : pathlib.Path
        ENVI radiance file whose header holds the wavelengths.
    file : pathlib.Path, optional
        Wavelength file with channel, wavelength, and FWHM columns to use
        instead of the radiance header. Must have one row per radiance channel.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Wavelength centers and FWHM. Values above 100 are taken to be nanometers
        and converted to microns.

    Raises
    ------
    ValueError
        If ``file`` does not have one row per radiance channel.

    Notes
    -----
    Without a FWHM in the header, the FWHM is the spacing of the first two
    channels.
    """
    meta = spectral_envi.open(envi_header(str(radiance))).metadata
    channels = len(meta["wavelength"])

    if file:
        _, wl, fwhm = np.loadtxt(file).T
        if len(wl) != channels:
            raise ValueError(
                f"Wavelength file {file} has {len(wl)} channels, which does not "
                f"match the {channels} channels of the radiance {radiance}"
            )
    else:
        Logger.info("Reading the wavelengths from the radiance header")
        wl = np.array(meta["wavelength"], dtype=float)
        fwhm = meta.get("fwhm", meta.get("FWHM"))
        if fwhm is not None:
            fwhm = np.array(fwhm, dtype=float)
        else:
            fwhm = np.full(wl.shape, wl[1] - wl[0])

    if wl[0] > 100:
        Logger.info("Wavelength units of nm inferred, converting to microns")
        wl, fwhm = wl / 1000, fwhm / 1000

    return wl, fwhm


def write_wavelengths(file: Path, wl: np.ndarray, fwhm: np.ndarray) -> None:
    """
    Write a wavelength file of channel index, wavelength, and FWHM columns.
    """
    data = np.column_stack([np.arange(len(wl)), wl, fwhm])
    np.savetxt(file, data, delimiter=" ")
