"""
Configuration loader utilities.

Configs are loaded either from a file on disk or from one of the templates
shipped under ``isofit/configs/templates/``, which are split into:

- ``bases/`` -- complete configs, typically inheriting ``default`` via ``^^``;
  see :func:`list_bases`
- ``mixins/`` -- overlays without ``^^``, grouped into files by purpose (e.g.
  ``sensors.yml``) and layered on top of a base; see :func:`list_mixins`

Both are referenced by the name of a top-level section in one of their files
(e.g. ``"default"``, ``"vswir"``, ``"emit"``). Each section carries a
``description`` of its purpose (see :func:`describe`), which is dropped when the
section is loaded.

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
from isofit.configs.sections.io import InputConfig
from isofit.configs.utils import interpolate
from isofit.data import env
from isofit.retrieval.utils.sensor import derive_fid

Logger = logging.getLogger(__name__)
Cache: dict[tuple[Path, Optional[str], bool], dict] = {}
ConfigsDir = Path(__file__).resolve().parent / "templates"
BasesDir = ConfigsDir / "bases"
MixinsDir = ConfigsDir / "mixins"


def index_sections(directory: Path) -> dict[str, Path]:
    """
    Index the top-level sections of the template files in ``directory``.

    Files are scanned in sorted order, so if two files define the same section
    the first one found wins.

    Parameters
    ----------
    directory : pathlib.Path
        Directory of template files.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of section name to the template file that defines it.
    """
    index = {}
    files = (p for p in directory.glob("*") if p.suffix in interpolate.Loaders)
    for path in sorted(files):
        for section in interpolate.read(path).keys():
            index.setdefault(section, path)

    return index


@functools.cache
def list_bases() -> dict[str, Path]:
    """
    Index the base configs available in :data:`BasesDir`.

    Every top-level section of every file in :data:`BasesDir` is a base config.
    The index is computed once and cached; the returned dict is shared, so do
    not mutate it.

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
    return index_sections(BasesDir)


@functools.cache
def list_mixins() -> dict[str, Path]:
    """
    Index the mixins available in :data:`MixinsDir`.

    Every top-level section of every file in :data:`MixinsDir` is a mixin, with
    each file grouping related mixins (e.g. ``sensors.yml``). The index is
    computed once and cached; the returned dict is shared, so do not mutate it.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of mixin name to the template file that defines it.

    Examples
    --------
    >>> from isofit.configs.loader import list_mixins
    >>> index = list_mixins()
    >>> index["emit"].name
    'sensors.yml'
    """
    return index_sections(MixinsDir)


def describe(ref: Union[str, Path]) -> Optional[str]:
    """
    Get the ``description`` of a config section.

    Parameters
    ----------
    ref : str or pathlib.Path
        The config reference; see :func:`resolve`.

    Returns
    -------
    str or None
        The section's own ``description``, or ``None`` if it has none.

    Examples
    --------
    >>> from isofit.configs.loader import describe
    >>> describe("tanager")
    'Tanager-1 instrument noise'
    """
    file, section = resolve(ref)

    data = interpolate.read(file)
    if section:
        data = data.get(section) or {}

    return data.get("description")


def resolve(ref: Union[str, Path]) -> tuple[Path, Optional[str]]:
    """
    Resolve a config reference to the file and top-level section it names.

    A reference is one of:

    - ``file`` -- an existing config file, loaded whole;
    - ``file:section`` -- a top-level section of a config file;
    - ``name`` -- a base config (see :func:`list_bases`) or a mixin (see
      :func:`list_mixins`).

    An existing file takes precedence over a base config of the same name,
    which in turn takes precedence over a mixin. The ``file:section`` split is
    made on the colon directly after a supported file extension, so colons
    elsewhere in the path are left untouched.

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
        If ``ref`` is neither an existing file, a known base config, nor a
        known mixin.

    Examples
    --------
    >>> from isofit.configs.loader import resolve
    >>> file, section = resolve("srtmnet")
    >>> file.name, section
    ('engines.yml', 'srtmnet')
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

    for index in (list_bases(), list_mixins()):
        if ref in index:
            return index[ref], ref

    msg = (
        f"Unknown config {ref!r}. Available base configs: {', '.join(list_bases())}."
        f" Available mixins: {', '.join(list_mixins())}"
    )
    Logger.error(msg)
    raise ValueError(msg)


def load_layer(ref: Union[str, Path], inherit: bool = True) -> dict:
    """
    Load a single config layer, upgraded from any legacy schema.

    The reference is resolved by :func:`resolve` and the selected file or
    section read without interpolation, then upgraded via
    :func:`isofit.configs.legacy.convert`. Layers read from the templates
    (bases and mixins) are stored in :data:`Cache`, while user files are re-read
    on every call.

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

    # Metadata of the template, not config; see `describe`
    data.pop("description", None)

    data = legacy.convert(data)

    # Only the templates are cached; user files may be rewritten
    if ident[0].is_relative_to(ConfigsDir):
        Cache[ident] = copy.deepcopy(data)

    return data


def merge_layer(data: dict, layer: dict) -> dict:
    """
    Deep-merge a config layer on top of ``data`` in place, switching engines.

    Layers are merged by :func:`isofit.configs.utils.interpolate.merge`, except
    that a layer selecting a different ``forward_model.atmosphere.engine``
    replaces the engine rather than merging into it, as each engine accepts
    different fields. Values ``data`` set on its engine beyond that engine's
    mixin defaults (e.g. the ``tc`` bases' ``template_file``) are carried over
    when the new engine has the same field and the layer leaves it unset.

    Parameters
    ----------
    data : dict
        The config to update.
    layer : dict
        The layer to merge on top, e.g. from :func:`load_layer`.

    Returns
    -------
    dict
        The same ``data``, updated.

    Examples
    --------
    >>> data = load_layer("pipeline")
    >>> _ = merge_layer(data, load_layer("modtran", inherit=False))
    >>> engine = data["forward_model"]["atmosphere"]["engine"]
    >>> engine["name"], "sixs_dir" in engine, engine["template_file"]
    ('modtran', False, '${output.dir}/config/${input.fid}_modtran_tpl.json')
    """
    old = engine_of(data)
    new = engine_of(layer)
    if not (old and new) or new.get("name") in (None, old.get("name")):
        return interpolate.merge(data, layer)

    # What `data` set on its old engine beyond that engine's mixin defaults
    defaults = {}
    if old.get("name") in list_mixins():
        defaults = engine_of(load_layer(old["name"], inherit=False)) or {}

    fields = legacy.engine_fields().get(legacy.engine_name(new["name"]), set())
    kept = {
        key: val
        for key, val in old.items()
        if key in fields and (key not in defaults or defaults[key] != val)
    }

    data["forward_model"]["atmosphere"]["engine"] = dict(kept)
    interpolate.merge(data, layer)

    # The carried values win over the new engine's unset (null) defaults
    engine = data["forward_model"]["atmosphere"]["engine"]
    for key, val in kept.items():
        if new.get(key) is None:
            engine[key] = val

    return data


def engine_of(data: dict) -> Optional[dict]:
    """The ``forward_model.atmosphere.engine`` mapping of ``data``, if any."""
    for key in ("forward_model", "atmosphere", "engine"):
        if not isinstance(data, dict):
            return None
        data = data.get(key)

    return data if isinstance(data, dict) else None


def load(
    key: Union[str, Path],
    section: Optional[str] = None,
    overrides: Optional[dict] = None,
    interp: bool = False,
    mixin: Optional[Union[str, Path, list[Union[str, Path]]]] = None,
) -> dict:
    """
    Load a config dict from a file or a named base config.

    ``key`` and each ``mixin`` are loaded as layers by :func:`load_layer`. Each
    ``mixin`` layer is then deep-merged on top of ``key``, in order, so later
    mixins win. A copy of the result then has
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
    mixin : str or pathlib.Path or list of them, optional
        Config references, in the same forms as ``key``, layered on top of it
        by :func:`merge_layer`, typically mixins (see :func:`list_mixins`).
        Only a mixin's own keys are applied: its ``^^`` inheritance is ignored,
        so a section of a base may also be layered without reapplying every
        ``default`` value over ``key``.

    Returns
    -------
    dict
        The loaded config, or its ``section``, in the current schema.

    Raises
    ------
    ValueError
        If ``key`` or a ``mixin`` is neither an existing file, a known base
        config, nor a known mixin.
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
    >>> config = load("default", mixin="srtmnet")
    >>> config["forward_model"]["atmosphere"]["engine"]["name"]
    'srtmnet'
    >>> config = load("pipeline", mixin=["emit", "empirical_line"])
    >>> config = load("vswir", mixin="sixs")
    >>> config["forward_model"]["atmosphere"]["engine"]["name"]
    'sixs'
    """
    data = load_layer(key)

    if isinstance(mixin, (str, Path)):
        mixin = [mixin]
    for ref in mixin or []:
        merge_layer(data, load_layer(ref, inherit=False))

    # Overrides are relative to the section
    sub = interpolate.get_dotted(data, section)
    interpolate.override(sub, overrides)

    # Before interpolation so references resolve by field name, e.g.
    # `${input.measured_radiance_file}` when the config sets `input.rdn`
    unalias(data)

    # Derived before interpolation so `${input.fid}` references resolve
    derive_fid(data)

    # Interpolations are against the full config
    if interp:
        # Temporarily add the ini env for easy interpolation
        data["env"] = {"modtran": None, **dict(env.items())}
        interpolate.interpolate(data)
        data.pop("env", None)

    return sub


def unalias(data: dict) -> None:
    """
    Rename aliased ``input`` keys to their field names, in place.

    An alias (e.g. ``rdn``) takes precedence over its field name
    (``measured_radiance_file``) when both are set, matching Pydantic.

    Examples
    --------
    >>> from isofit.configs.loader import unalias
    >>> data = {"input": {"rdn": "a", "measured_radiance_file": "b"}}
    >>> unalias(data)
    >>> data
    {'input': {'measured_radiance_file': 'a'}}
    """
    inputs = data.get("input")
    if not isinstance(inputs, dict):
        return

    for name, field in InputConfig.model_fields.items():
        if field.alias in inputs:
            inputs[name] = inputs.pop(field.alias)


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
    ``path``. Empty strings, absolute paths, and values starting with a
    ``${...}`` reference are left unchanged.

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
            and val
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
