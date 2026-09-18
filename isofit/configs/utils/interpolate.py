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
    Load a configuration from a YAML file with optional interpolation.

    Reads a YAML configuration file, optionally extracts a specific section,
    applies command-line overrides from a Click context, and performs variable
    interpolation to resolve ${...} references.

    Parameters
    ----------
    config : str
        File path to the configuration YAML file.
    section : str, optional
        If provided, returns only this subsection of the config instead of
        the full configuration. By default None.
    ctx : click.Context, optional
        Click context object containing command-line arguments for patching
        the configuration. By default None.
    interp : bool, optional
        If True, calls interpolate() on the config to resolve ${...} variable
        references before returning. By default True.

    Returns
    -------
    Box
        Loaded configuration as a Box object with dot-notation access.

    Examples
    --------
    >>> config = load("config.yaml")
    >>> print(config.database.host)
    localhost
    >>> config = load("config.yaml", section="database")
    >>> print(config.host)
    localhost

    See Also
    --------
    patch : Apply command-line overrides to configuration
    interpolate : Resolve ${...} variable references
    """
    config = Box.from_yaml(filename=config, default_box=True)

    if section:
        config = config[section]

    if ctx:
        patch(config, ctx)

    if interp:
        interpolate(config)

    return config


def patch(box: Box, ctx: click.Context) -> Box:
    """
    Patch the config in place with dotted ``--key value`` options from the CLI.

    Scans ``ctx.args`` for ``--dotted.key value`` pairs, parses each value as a
    Python literal (falling back to a string), and merges them into ``box`` using
    Box dot-notation. This allows command-line arguments to override nested
    configuration values.

    Parameters
    ----------
    box : box.Box
        The loaded config to override. Modified in place.
    ctx : click.Context
        Click context whose ``args`` attribute contains the extra ``--key value``
        overrides to apply.

    Returns
    -------
    box.Box
        The same ``box`` object, updated with the CLI overrides.

    Examples
    --------
    Command-line usage:

    .. code-block:: bash

        $ python script.py --database.host localhost --database.port 5432

    This would override:

    >>> # config["database"]["host"] = "localhost"
    >>> # config["database"]["port"] = 5432

    Notes
    -----
    Values are parsed using ast.literal_eval() to convert strings like
    '["band", 10]' into actual Python lists. If parsing fails, the value
    is kept as a string.

    See Also
    --------
    load : Load configuration with automatic patching
    """
    if isinstance(ctx, click.Context):
        ctx = ctx.args

    # Convert dot notation to dict
    conv = Box(default_box=True, box_dots=True)

    i = 0
    while i < len(ctx):
        arg = ctx[i]

        if arg.startswith("--"):
            key = arg[2:]
            val = ctx[i + 1]
            try:
                val = ast.literal_eval(val)
            except:
                Logger.warning(
                    f"Failed to parse the value and will default as string: --{key} {val}"
                )

            Logger.debug(f"Overriding {key} with {val!r}")
            conv[key] = val

            i += 2
        else:
            i += 1

    # Override config with new converted values
    if isinstance(box, dict):
        box = Box(box, default_box=True)
    box.merge_update(conv)

    return box


def interp(val: str, rel: Box, full: Box) -> Any:
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

    Returns
    -------
    Any
        The interpolated value; a literal (int/list/...) when it parses as one,
        otherwise the substituted string (or ``val`` unchanged if it had no refs).
    """
    if matches := Interp.findall(val):
        for key in matches:
            if key.startswith("."):
                Logger.debug("Using relative pathing")
                ref = rel
            else:
                Logger.debug("Using full pathing")
                ref = full

            new = ref[key]
            if isinstance(new, str) and "${" in new:
                new = interp(new, rel, full)  # TODO: Recursion guard

            fmt = "${" + key + "}"
            val = val.replace(fmt, str(new))
            Logger.debug(f"Replaced {fmt!r} with {new!r}")
        try:
            val = ast.literal_eval(val)
        except:
            pass
        Logger.debug(f"New value: {val!r}")

    return val


def interpolate(
    box: Union[Box, BoxList],
    orig: Optional[Box] = None,
    rel: Optional[Box] = None,
) -> None:
    """
    Recursively interpolate every ``${...}`` reference in a config tree in place.

    Walks ``box`` (a Box or BoxList), replacing each string value via :func:`interp`.
    ``orig`` is the full config used for absolute references; ``rel`` is the current
    subsection used for relative ones. Both default to ``box`` itself on the first
    call and are threaded down as the walk descends into subsections.

    Parameters
    ----------
    box : box.Box or box.BoxList
        The config node to interpolate; mutated in place.
    orig : box.Box, optional
        Full config for absolute references. Defaults to ``box`` on the first call.
    rel : box.Box, optional
        Subsection for relative references. Defaults to ``orig``.
    """
    if isinstance(box, dict):
        box = Box(box, default_box=True)

    if orig is None:
        orig = Box(box, box_dots=True, default_box=True)

    if rel is None:
        rel = orig

    if isinstance(box, BoxList):
        items = enumerate(box)
    else:
        rel = Box(box, box_dots=True, default_box=True)
        items = box.items()

    for key, val in items:
        if isinstance(val, (Box, BoxList)):
            box[key] = interpolate(val, orig, rel)
        elif isinstance(val, str):
            box[key] = interp(val, rel, orig)

    return box
