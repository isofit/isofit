"""
Configuration loader utilities.

Configs are loaded either from a file on disk or from one of the base configs
shipped under ``isofit/configs/templates/``. Base configs are referenced by the
name of a top-level section in one of those files (e.g. ``"default"``,
``"vswir"``, ``"modtran"``); see :func:`list_bases` for the full index.

Loading resolves ``^^`` inheritance via :mod:`isofit.configs.utils.interpolate`,
upgrades the result from any legacy schema via :func:`isofit.configs.legacy.convert`,
then applies overrides and optionally ``${...}`` interpolation. The ISOFIT
environment (:data:`isofit.data.env`) is exposed under the ``env`` key during
interpolation so configs may reference e.g. ``${env.srtmnet}``.
"""

import copy
import functools
import logging
from pathlib import Path
from typing import Optional, Union

from isofit.configs import legacy
from isofit.configs.config import Config
from isofit.configs.utils import interpolate
from isofit.data import env

Logger = logging.getLogger(__name__)
Cache: dict[tuple[Union[str, Path], Optional[str]], dict] = {}
ConfigsDir = Path(__file__).resolve().parent / "templates"


@functools.cache
def list_bases() -> dict[str, Path]:
    """
    Index the base configs available in :data:`ConfigsDir`.

    Every top-level section of every template file is a base config. Files are
    scanned in sorted order, so if two files define the same section the first
    one found wins. The index is computed once and cached; the returned dict is
    shared, so do not mutate it.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of base config name to the template file that defines it.

    Examples
    --------
    >>> from isofit.configs.loader import list_bases
    >>> index = list_bases()
    >>> index["vswir"].name
    'vswir.yml'
    """
    index = {}
    files = (p for p in ConfigsDir.glob("*") if p.suffix in {".yaml", ".yml", ".json"})
    for path in sorted(files):
        for section in interpolate.read(path).keys():
            index.setdefault(section, path)

    return index


def load(
    key: Union[str, Path],
    section: Optional[str] = None,
    overrides: Optional[dict] = None,
    interp: bool = False,
) -> dict:
    """
    Load a config dict from a file or a named base config.

    The file (or, for a base config, its section) is read with
    :func:`isofit.configs.utils.interpolate.load` (resolving ``^^`` inheritance,
    without interpolation) and upgraded from any legacy schema via
    :func:`isofit.configs.legacy.convert`; for base configs that result is
    stored in :data:`Cache` under ``(file, base name)``, while files given by
    path are re-read on every call. A copy of it then has
    ``overrides`` applied to ``section``, and if ``interp`` is set its
    ``${...}`` references are resolved against the full config, so a section
    may reference values outside itself (e.g. ``${output.dir}`` from within
    ``forward_model``). During interpolation the ISOFIT environment is
    temporarily exposed under ``env`` so configs may reference e.g.
    ``${env.srtmnet}``. Only ``section`` is returned.

    Parameters
    ----------
    key : str or pathlib.Path
        Either a path to a config file (.yml, .yaml, or .json) or the name of a
        base config (see :func:`list_bases`). An existing file takes precedence
        over a base config of the same name.
    section : str, optional
        Dotted path of the subsection to return, e.g. ``"forward_model.surface"``
        (see :func:`isofit.configs.utils.interpolate.get_dotted`). ``None`` or
        ``""`` returns the full config.
    overrides : dict, optional
        Dotted-key overrides, ``{"dotted.key": value}``, relative to ``section``
        rather than the full config. A mapping is deep-merged into an existing
        mapping; any other value replaces the existing one outright (see
        :func:`isofit.configs.utils.interpolate.override`).
    interp : bool, default=False
        Resolve ``${...}`` references before returning.

    Returns
    -------
    dict
        The loaded config, or its ``section``, in the current schema.

    Raises
    ------
    ValueError
        If ``key`` is neither an existing file nor a known base config.
    KeyError
        If ``section`` is not in the loaded config, or a ``${...}`` reference
        cannot be resolved.

    Examples
    --------
    >>> from isofit.configs.loader import load
    >>> config = load("vswir", interp=True)
    >>> impl = load("vswir", section="implementation", overrides={"n_cores": 4})
    >>> impl["n_cores"]
    4
    """
    if Path(key).is_file():
        file = key
        key = None
    elif key in (index := list_bases()):
        file = index[key]
    else:
        valid = ", ".join(index)
        msg = f"Unknown base config {key!r}. Available base configs: {valid}"
        Logger.error(msg)
        raise ValueError(msg)

    if (file, key) in Cache:
        data = copy.deepcopy(Cache[(file, key)])
    else:
        # Loading by section is what resolves that section's `^^` inheritance
        data = interpolate.load(file, section=key, interp=False)
        data = legacy.convert(data)

        # Only the templates are cached; user files may be rewritten
        if key:
            Cache[(file, key)] = copy.deepcopy(data)

    # Overrides are relative to the section
    sub = interpolate.get_dotted(data, section)
    interpolate.override(sub, overrides)

    # Interpolations are against the full config
    if interp:
        # Temporarily add the ini env for easy interpolation
        data["env"] = {"modtran": None, **dict(env.items())}
        interpolate.interpolate(data)
        data.pop("env", None)

    return sub


def create_new_config(file: Union[str, Path], **kwargs) -> Config:
    """
    Load a config and validate it into a :class:`~isofit.configs.Config`.

    Legacy loading function, to be deprecated in the config refactor part 2.

    When ``file`` is a path, relative paths in the config (values of keys
    containing ``file``, ``directory``, or ``path``) are expanded against the
    config file's directory, not the current working directory. Values that
    are ``${...}`` references are left for interpolation.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to a config file or the name of a base config; see :func:`load`.
    **kwargs
        Passed through to :func:`load` (``section``, ``overrides``, ``interp``).

    Returns
    -------
    Config
        The validated config.

    Raises
    ------
    ValueError
        If ``file`` is neither an existing file nor a known base config.
    pydantic.ValidationError
        If the loaded config fails validation.

    Examples
    --------
    >>> from isofit.configs import create_new_config
    >>> config = create_new_config("config.yaml")
    """
    data = load(file, **kwargs)

    if Path(file).is_file():
        expand_paths(data, Path(file).resolve().parent)

    return Config.model_validate(data)


def expand_paths(data: Union[dict, list], root: Path) -> None:
    """
    Expand relative paths in a config tree against ``root``, in place.

    Applies to string values of keys containing ``file``, ``directory``, or
    ``path``. Absolute paths and values starting with a ``${...}`` reference
    are left unchanged.

    Parameters
    ----------
    data : dict or list
        The config node to expand; mutated in place.
    root : pathlib.Path
        Directory relative paths are expanded against.
    """
    items = enumerate(data) if isinstance(data, list) else data.items()
    for key, val in items:
        if isinstance(val, (dict, list)):
            expand_paths(val, root)
        elif (
            isinstance(key, str)
            and isinstance(val, str)
            and any(tag in key for tag in ("file", "directory", "path"))
            and not val.startswith("${")
            and not Path(val).is_absolute()
        ):
            data[key] = str(root / val)
