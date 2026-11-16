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

from pydantic import ValidationError

from isofit.configs import legacy
from isofit.configs.config import Config
from isofit.configs.utils import interpolate
from isofit.data import env
from isofit.retrieval.sensor import derive_fid

Logger = logging.getLogger(__name__)
Cache: dict[tuple[Path, Optional[str], bool], dict] = {}
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


def resolve(ref: Union[str, Path]) -> tuple[Path, Optional[str]]:
    """
    Resolve a config reference to the file and top-level section it names.

    A reference is one of:

    - ``file`` -- an existing config file, loaded whole;
    - ``file:section`` -- a top-level section of a config file;
    - ``name`` -- a base config (see :func:`list_bases`).

    An existing file takes precedence over a base config of the same name. The
    ``file:section`` split is made on the colon directly after a supported file
    extension, so colons elsewhere in the path are left untouched.

    Parameters
    ----------
    ref : str or pathlib.Path
        The config reference.

    Returns
    -------
    tuple[pathlib.Path, str or None]
        The config file and the section within it, or ``None`` for the whole
        file.

    Raises
    ------
    ValueError
        If ``ref`` is neither an existing file nor a known base config.

    Examples
    --------
    >>> from isofit.configs.loader import resolve
    >>> file, section = resolve("srtmnet")
    >>> file.name, section
    ('default.yml', 'srtmnet')
    """
    if Path(ref).is_file():
        return Path(ref), None

    ref = str(ref)
    for ext in interpolate.Loaders:
        marker = ref.find(ext + ":")
        if marker != -1:
            split = marker + len(ext)
            file, section = Path(ref[:split]), ref[split + 1 :]
            if not file.is_file():
                raise ValueError(f"Config file not found: {str(file)!r} (from {ref!r})")
            return file, section or None

    if ref in (index := list_bases()):
        return index[ref], ref

    valid = ", ".join(index)
    msg = f"Unknown base config {ref!r}. Available base configs: {valid}"
    Logger.error(msg)
    raise ValueError(msg)


def load_layer(ref: Union[str, Path], inherit: bool = True) -> dict:
    """
    Load a single config layer, upgraded from any legacy schema.

    The reference is resolved by :func:`resolve` and the selected file or
    section read without interpolation, then upgraded via
    :func:`isofit.configs.legacy.convert`. Layers read from the base config
    templates are stored in :data:`Cache`, while user files are re-read on every
    call.

    Parameters
    ----------
    ref : str or pathlib.Path
        The config reference; see :func:`resolve`.
    inherit : bool, default=True
        Resolve the layer's ``^^`` inheritance. If ``False`` its ``^^`` key is
        dropped instead, leaving only the layer's own keys.

    Returns
    -------
    dict
        A copy of the layer, safe to mutate.

    Raises
    ------
    ValueError
        If ``ref`` is neither an existing file nor a known base config.
    KeyError
        If ``ref`` names a section not in its file.
    """
    file, section = resolve(ref)

    ident = (file.resolve(), section, inherit)
    if ident in Cache:
        return copy.deepcopy(Cache[ident])

    if inherit:
        # Loading by section is what resolves that section's `^^` inheritance
        data = interpolate.load(file, section=section, interp=False)
    else:
        data = interpolate.read(file)
        if section:
            if section not in data:
                raise KeyError(f"Section {section!r} not found in {str(file)!r}")
            data = data[section]
        data = copy.deepcopy(data)
        data.pop("^^", None)

    data = legacy.convert(data)

    # Only the templates are cached; user files may be rewritten
    if ident[0].parent == ConfigsDir:
        Cache[ident] = copy.deepcopy(data)

    return data


def load(
    key: Union[str, Path],
    section: Optional[str] = None,
    overrides: Optional[dict] = None,
    interp: bool = False,
    patch: Optional[Union[str, Path, list[Union[str, Path]]]] = None,
) -> dict:
    """
    Load a config dict from a file or a named base config.

    ``key`` and each ``patch`` are loaded as layers by :func:`load_layer`. Each
    ``patch`` layer is then deep-merged on top of ``key``, in order, so later
    patches win. A copy of the result then has
    ``overrides`` applied to ``section``, and if ``interp`` is set its
    ``${...}`` references are resolved against the full config, so a section
    may reference values outside itself (e.g. ``${output.dir}`` from within
    ``forward_model``). During interpolation the ISOFIT environment is
    temporarily exposed under ``env`` so configs may reference e.g.
    ``${env.srtmnet}``. Only ``section`` is returned.

    Parameters
    ----------
    key : str or pathlib.Path
        A config reference in any form accepted by :func:`resolve`: a base
        config name, a file path, or ``file:section``.
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
    patch : str or pathlib.Path or list of them, optional
        Config references, in the same forms as ``key``, layered on top of it.
        Only a patch's own keys are applied: its ``^^`` inheritance is ignored,
        so e.g. ``patch="srtmnet"`` adds the sRTMnet engine without also
        reapplying every ``default`` value over ``key``.

    Returns
    -------
    dict
        The loaded config, or its ``section``, in the current schema.

    Raises
    ------
    ValueError
        If ``key`` or a ``patch`` is neither an existing file nor a known base
        config.
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
    >>> config = load("default", patch="srtmnet")
    >>> config["forward_model"]["atmosphere"]["engine"]["name"]
    'srtmnet'
    """
    data = load_layer(key)

    if isinstance(patch, (str, Path)):
        patch = [patch]
    for ref in patch or []:
        interpolate.merge(data, load_layer(ref, inherit=False))

    # Overrides are relative to the section
    sub = interpolate.get_dotted(data, section)
    interpolate.override(sub, overrides)

    # Derived before interpolation so `${input.fid}` references resolve
    derive_fid(data)

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


def make_config(data, safe=True, rich=True):
    """ """
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        Logger.error(f"The configuration has {e.error_count()} validation error(s)")
        if rich:
            from isofit.configs.utils.console import errors_table
            from isofit.retrieval import Console

            Console.print(errors_table(e, Config))
        elif not safe:
            raise
