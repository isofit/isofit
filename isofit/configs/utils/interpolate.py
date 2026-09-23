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
from typing import Any, Optional, Union

import click
from box import Box, BoxList

Logger = logging.getLogger(__name__)
Interp = re.compile(r"\${([^}]+)}")


def load(
    config: str,
    section: Optional[str] = None,
    ctx: Optional[click.Context] = None,
    interp: bool = True,
) -> Box:
    """
    Loads a config from a yaml file

    Parameters
    ----------
    config : str
        File path to the config.yml
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
    box = full = Box.from_yaml(filename=config, default_box=True)

    if section:
        box = full[section]

    if "^^" in box:
        box = patch(box, full)

    if ctx:
        box = override(box, ctx)

    if interp:
        box = interpolate(box)

    return box


def patch(box: Box, full: Box, _seen: Optional[set] = None) -> Box:
    """
    Merge one or more referenced sections into ``box`` in place.

    Consumes the ``^^`` key from ``box``, whose value names either a single section
    (str) or several sections (BoxList) elsewhere in ``full``. The named sections are
    treated as inherited defaults: they are applied lowest-priority first, in the
    order listed, and ``box``'s own keys always win on conflict. Referenced sections
    are resolved recursively, so a section that itself declares ``^^`` has its own
    ancestors folded in first.

    For example ``A: {^^: [B, C]}`` with ``B: {^^: D}`` yields the precedence
    ``D < B < C < A`` (``A``'s own keys highest, ``D`` lowest).

    Parameters
    ----------
    box : box.Box
        The config subsection to patch; mutated in place. Its ``^^`` key is removed.
    full : box.Box
        The full config used to resolve the referenced section names.
    _seen : set, optional
        Internal set of section names currently being resolved on this recursion
        chain, used to detect circular ``^^`` references. Should not be passed by
        callers.

    Returns
    -------
    box.Box
        The same ``box``, updated with the merged sections.
    """
    sects = box.pop("^^")

    if isinstance(sects, str):
        sects = [sects]

    # Accumulate referenced sections lowest-priority first (list order), then let
    # box's own keys win by merging them in last.
    merged = Box(default_box=True)
    for sect in sects:
        if sect not in full:
            raise KeyError(f"^^ references an unknown section: {sect!r}")

        if _seen and sect in _seen:
            raise ValueError(f"Circular ^^ reference detected for section: {sect!r}")

        ref = Box(full[sect], default_box=True)
        if "^^" in ref:
            ref = patch(ref, full, (_seen or set()) | {sect})

        merged.merge_update(ref)

    merged.merge_update(box)

    box.clear()
    box.merge_update(merged)

    return box


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
