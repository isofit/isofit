"""
Configuration interpolation utilities for variable substitution.

This module provides utilities for loading configuration files with variable
interpolation support. It allows using ${key} or ${.key} syntax to reference
other values within the configuration, and supports command-line overrides
through Click contexts.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

import click
import yaml
from box import Box, BoxList

Logger = logging.getLogger(__name__)
Interp = re.compile(r"\${([^}]+)}")

# Supported config file extensions, mapped to the Box loader for that format.
Loaders = {
    ".yml": Box.from_yaml,
    ".yaml": Box.from_yaml,
    ".json": Box.from_json,
}


def read(config: Union[str, Path]) -> Box:
    """
    Load a config file into a Box, dispatching on its extension.

    Both YAML (``.yml`` / ``.yaml``) and JSON (``.json``) are supported.

    Parameters
    ----------
    config : str or pathlib.Path
        File path to the config to read.

    Returns
    -------
    box.Box
        The loaded configuration.

    Raises
    ------
    ValueError
        If the file's extension is not a supported config format.
    """
    suffix = Path(config).suffix.lower()
    if suffix not in Loaders:
        supported = ", ".join(sorted(Loaders))
        raise ValueError(
            f"Unsupported config file extension {suffix!r} for {str(config)!r}; "
            f"expected one of: {supported}"
        )
    return Loaders[suffix](filename=str(config), default_box=True)


def load(
    config: str,
    section: Optional[str] = None,
    ctx: Optional[click.Context] = None,
    interp: bool = True,
) -> Box:
    """
    Loads a config from a yaml or json file

    Parameters
    ----------
    config : str
        File path to the config (.yml, .yaml, or .json)
    section : str
        Returns only this subsection of the config
    ctx : obj, default=None
        Click context object
    interp : bool, default=True
        Calls interpolate on the config before returning

    Returns
    -------
    config : Box
        Loaded configuration using Box
    """
    box = full = read(config)

    if section:
        box = full[section]

    if "^^" in box:
        path = Path(config).resolve()
        box = patch(box, full, base_dir=path.parent, cache={path: full})

    if ctx:
        box = override(box, ctx)

    if interp:
        box = interpolate(box)

    return box


def patch(
    box: Box,
    full: Box,
    _seen: Optional[set] = None,
    base_dir: Optional[Path] = None,
    cache: Optional[dict] = None,
) -> Box:
    """
    Merge one or more referenced sections into ``box`` in place.

    Consumes the ``^^`` key from ``box``, whose value names either a single section
    (str) or several sections (BoxList). Each name is treated as an inherited
    default; they are applied lowest-priority first, in the order listed, and
    ``box``'s own keys always win on conflict. Referenced sections are resolved
    recursively, so a section that itself declares ``^^`` has its own ancestors
    folded in first.

    For example ``A: {^^: [B, C]}`` with ``B: {^^: D}`` yields the precedence
    ``D < B < C < A`` (``A``'s own keys highest, ``D`` lowest).

    References may point either at a section in the same file (``^^: default``) or
    at a section in another file using ``file:section`` syntax (``^^:
    bases/default.yml:srtmnet``). The referenced file may be YAML or JSON. The
    file portion is resolved relative to ``base_dir`` (the directory of the file
    being loaded); a bare ``file`` with no ``:section`` inherits that file's
    top-level mapping.

    Parameters
    ----------
    box : box.Box
        The config subsection to patch; mutated in place. Its ``^^`` key is removed.
    full : box.Box
        The full config used to resolve same-file section names.
    _seen : set, optional
        Internal set of (file, section) keys currently being resolved on this
        recursion chain, used to detect circular ``^^`` references. Should not be
        passed by callers.
    base_dir : pathlib.Path, optional
        Directory used to resolve the file portion of cross-file references.
        Defaults to the current working directory.
    cache : dict, optional
        Internal mapping of resolved file path to its loaded Box, so each
        referenced file is read from disk only once per ``patch`` tree.

    Returns
    -------
    box.Box
        The same ``box``, updated with the merged sections.
    """
    if isinstance(box, dict):
        box = Box(box)

    if base_dir is None:
        base_dir = Path.cwd()
    if cache is None:
        cache = {}
    _seen = _seen or set()

    sects = box.pop("^^", [])
    if isinstance(sects, str):
        sects = [sects]

    # Accumulate referenced sections lowest-priority first (list order), then let
    # box's own keys win by merging them in last.
    merged = Box(default_box=True)
    for sect in sects:
        # Resolve the reference to (its full config, the subsection Box, the
        # base_dir for *its* own ^^ refs, and a hashable identity for cycle
        # detection). Cross-file refs carry a supported file extension, either
        # bare (`path.yml`) or with a `:section` suffix (`path.yml:section`).
        if any(ext in sect for ext in Loaders):
            ref_full, ref, ref_dir, ident = resolve_file(sect, base_dir, cache)
        else:
            if sect not in full:
                raise KeyError(f"^^ references an unknown section: {sect!r}")
            ref_full = full
            ref = Box(full[sect], default_box=True)
            ref_dir = base_dir
            ident = (None, sect)

        if ident in _seen:
            raise ValueError(f"Circular ^^ reference detected for section: {sect!r}")

        if "^^" in ref:
            ref = patch(ref, ref_full, _seen | {ident}, base_dir=ref_dir, cache=cache)

        merged.merge_update(ref)

    merged.merge_update(box)

    box.clear()
    box.merge_update(merged)

    return box


def resolve_file(sect: str, base_dir: Path, cache: dict) -> tuple:
    """
    Resolve a cross-file ``^^`` reference of the form ``path[:section]``.

    The path is taken relative to ``base_dir`` (the directory of the file that
    declared the reference) and loaded at most once per ``patch`` tree via
    ``cache``. With no ``:section`` suffix the referenced file's top-level
    mapping is inherited; otherwise the named section within it is used.

    The split is made on the colon that immediately follows a supported file
    extension (``.yml`` / ``.yaml`` / ``.json``), so colons elsewhere in the
    path are left untouched.

    Parameters
    ----------
    sect : str
        The raw reference string, e.g. ``bases/default.yml:srtmnet`` or
        ``../common.json``.
    base_dir : pathlib.Path
        Directory used to resolve ``path`` relative references.
    cache : dict
        Mapping of resolved file path to loaded Box, mutated in place.

    Returns
    -------
    tuple
        ``(ref_full, ref, ref_dir, ident)`` where ``ref_full`` is the referenced
        file's full config (for resolving that file's own same-file ``^^`` refs),
        ``ref`` is the selected subsection Box, ``ref_dir`` is the directory of
        the referenced file (base for its cross-file refs), and ``ident`` is a
        hashable ``(path, section)`` identity for cycle detection.
    """
    # Split on the colon directly after the file extension; a bare path (no
    # trailing `:section`) inherits the referenced file's top-level mapping.
    for ext in Loaders:
        marker = sect.find(ext + ":")
        if marker != -1:
            split = marker + len(ext)
            path, name = sect[:split], sect[split + 1 :]
            break
    else:
        path, name = sect, None
    name = name or None

    abspath = (base_dir / path).resolve()
    if abspath not in cache:
        if not abspath.exists():
            raise FileNotFoundError(
                f"^^ references a file that does not exist: {str(abspath)!r} (from {sect!r})"
            )
        cache[abspath] = read(abspath)

    ref_full = cache[abspath]

    if name is None:
        ref = Box(ref_full, default_box=True)
    else:
        if name not in ref_full:
            raise KeyError(
                f"^^ references an unknown section {name!r} in file {str(abspath)!r}"
            )
        ref = Box(ref_full[name], default_box=True)

    ref_dir = abspath.parent
    return ref_full, ref, ref_dir, (abspath, name)


def override(box: Box, ctx: Union[click.Context, list[str]]) -> Box:
    """
    Patch the config in place with dotted ``--key value`` options from the CLI.

    Scans the extra CLI args for ``--dotted.key value`` pairs, parses each value as a
    Python literal (falling back to a string), and merges them into ``box`` using Box
    dot-notation, so e.g. ``--tetrun.args '["band", 10]'`` overrides a nested key. A
    bare ``--key`` with no following value (at the end of the args or immediately
    before another ``--flag``) is treated as a boolean switch and set to ``True``.

    Parameters
    ----------
    box : box.Box
        The loaded config to override; mutated in place.
    ctx : click.Context or list of str
        Click context whose ``args`` carry the extra ``--key value`` overrides, or the
        list of extra args directly.

    Returns
    -------
    box.Box
        The same ``box``, updated with the overrides.
    """
    if isinstance(ctx, click.Context):
        args = ctx.args
    else:
        args = ctx

    # Assign overrides directly with box_dots so dotted keys resolve into nested
    # sections. merge_update is deliberately avoided: it silently drops None values
    # when merging into an existing section, so `--key None` would be a no-op.
    if isinstance(box, dict):
        box = Box(box, default_box=True, box_dots=True)

    i = 0
    while i < len(args):
        arg = args[i]

        if arg.startswith("--"):
            key = arg[2:]

            # A bare flag (end of args, or immediately followed by another --flag)
            # is a boolean switch and resolves to True, e.g. `--key`
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                Logger.debug(f"Setting {key} as True")
                box[key] = True
                i += 1
                continue

            val = args[i + 1]
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # A bare string is a valid value, so a parse failure is only worth
                # flagging when the value looks like an intended literal (list, dict,
                # tuple, quoted string, or number) but is malformed.
                if val[:1] in "[{('\"" or val[:1].isdigit() or val[:1] == "-":
                    Logger.warning(
                        f"Could not parse --{key} {val!r} as a Python literal; using it as a plain string"
                    )
                else:
                    Logger.debug(f"Using --{key} {val!r} as a plain string")

            Logger.debug(f"Overriding {key} with {val!r}")
            box[key] = val

            i += 2
        else:
            i += 1

    return box


def interp(val: str, rel: Box, full: Box, _seen: Optional[set] = None) -> Any:
    """
    Interpolate ``${...}`` references in a single string against the config.

    Format: ``${.key}`` or ``${key}``. A leading dot makes the key relative to the
    subsection the original value lives in (``rel``); otherwise it resolves from the
    top of the config (``full``). After substitution the result is parsed as a Python
    literal when possible, so a fully-substituted string can become e.g. an int/list.

    Parameters
    ----------
    val : str
        The value to interpolate.
    rel : box.Box
        The subsection used to resolve relative (``${.key}``) references.
    full : box.Box
        The full config used to resolve absolute (``${key}``) references.
    _seen : set, optional
        Internal set of keys currently being resolved on this recursion chain,
        used to detect circular references. Should not be passed by callers.

    Returns
    -------
    Any
        The interpolated value; a literal (int/list/...) when it parses as one,
        otherwise the substituted string (or ``val`` unchanged if it had no refs).
    """
    if matches := Interp.findall(val):
        if _seen is None:
            _seen = set()

        for key in matches:
            match = "${" + key + "}"

            if key in _seen:
                msg = f"Circular interpolation reference detected for {match!r}"
                Logger.error(msg)
                raise ValueError(msg)

            if key.startswith("."):
                Logger.debug("Using relative pathing")
                ref = rel
                lookup = key[1:]
            else:
                Logger.debug("Using full pathing")
                ref = full
                lookup = key

            if lookup not in ref:
                msg = f"Interpolation reference {match!r} not found in config"
                Logger.error(msg)
                raise KeyError(msg)

            new = ref[lookup]
            if isinstance(new, str) and "${" in new:
                new = interp(new, rel, full, _seen | {key})

            val = val.replace(match, str(new))
            Logger.debug(f"Replaced {match!r} with {new!r}")
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
        Logger.debug(f"New value: {val!r}")

    return val


def interpolate(
    box: Union[Box, BoxList],
    full: Optional[Box] = None,
    rel: Optional[Box] = None,
) -> Union[Box, BoxList]:
    """
    Recursively interpolate every ``${...}`` reference in a config tree in place.

    Walks ``box`` (a Box or BoxList), replacing each string value via :func:`interp`.
    ``full`` is the full config used for absolute references; ``rel`` is the current
    subsection used for relative ones. Both default to ``box`` itself on the first
    call and are threaded down as the walk descends into subsections.

    Parameters
    ----------
    box : box.Box or box.BoxList
        The config node to interpolate; mutated in place.
    full : box.Box, optional
        Full config for absolute references. Defaults to ``box`` on the first call.
    rel : box.Box, optional
        Subsection for relative references. Defaults to ``full``.

    Returns
    -------
    box.Box or box.BoxList
        The same ``box``, with all ``${...}`` references interpolated.
    """
    if isinstance(box, dict):
        box = Box(box, default_box=True)

    if full is None:
        full = Box(box, box_dots=True, default_box=True)

    if rel is None:
        rel = full

    if isinstance(box, BoxList):
        items = enumerate(box)
    else:
        rel = Box(box, box_dots=True, default_box=True)
        items = box.items()

    for key, val in items:
        if isinstance(val, (Box, BoxList)):
            box[key] = interpolate(val, full, rel)
        elif isinstance(val, str):
            box[key] = interp(val, rel, full)

    return box


@click.group(name="config", invoke_without_command=True, no_args_is_help=True)
def cli():
    """
    Utility functions for configuration files
    """
    logging.basicConfig(level=logging.DEBUG)


CS = dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
)
Config = click.argument("config")
Section = click.option(
    "-s", "--section", help="Subsection of the yaml to load rather than the whole file"
)
NoFlow = click.option(
    "--noflow",
    is_flag=True,
    help="Disables the YAML flow style which condenses lists to [a, b] notation",
)


def _format_yaml(box, noflow):
    if noflow:
        return yaml.dump(
            box.to_dict(), default_flow_style=None, sort_keys=False, width=120
        )
    return box.to_yaml()


@cli.command(context_settings=CS)
@click.pass_context
@Config
@Section
@NoFlow
@click.option("-v", "--validate", is_flag=True, help="Validate the config")
def preview(ctx: click.Context, noflow=False, validate=False, **kwargs: Any) -> None:
    """
    Preview the final, interpolated configuration.

    Loads and processes the configuration file (with interpolation and
    overrides) and displays it in YAML format without executing any
    pipeline stages. Useful for debugging config issues.
    """
    box = load(ctx=ctx, **kwargs)
    yml = _format_yaml(box, noflow)
    Logger.info(yml)

    if validate:
        Logger.info(
            "Validating the config. If there are no errors, nothing will be printed."
        )

        from isofit.configs import load_config_dict

        load_config_dict(box.to_dict())


@cli.command(context_settings=CS)
@click.pass_context
@Config
@Section
@NoFlow
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(writable=True, path_type=Path),
    help="File to write the new configuration to",
)
def copy(ctx: click.Context, output, noflow=False, **kwargs: Any) -> None:
    """
    Copies an existing configuration to a new file
    """
    box = load(ctx=ctx, **kwargs)

    if output.suffix in (".yml", ".yaml"):
        data = _format_yaml(box, noflow)
    elif output.suffix == ".json":
        data = box.to_json(indent=4)
    else:
        raise TypeError("Unsupported file extension, expected either .yaml or .json")

    output.write_text(data)
    Logger.info(f"Wrote to {output}")


@cli.command(context_settings=CS)
@click.pass_context
@Config
@Section
def validate(ctx: click.Context, **kwargs: Any) -> None:
    """
    Validates a configuration
    """
    box = load(ctx=ctx, **kwargs)

    Logger.info(
        "Validating the config. If there are no errors, nothing will be printed."
    )

    from isofit.configs import load_config_dict

    load_config_dict(box.to_dict())
