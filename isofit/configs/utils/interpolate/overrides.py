"""
Stage 2 of the pipeline: dotted-key overrides.

Overrides arrive as a ``click.Context``, a list of ``--dotted.key value`` CLI
tokens, or a ``{dotted.key: value}`` dict. :func:`parse_overrides` normalizes
them to a dict and :func:`override` applies that dict to the config.
"""

import ast
import logging
from typing import Union

import click

from .common import set_dotted

Logger = logging.getLogger(__name__)


def _parse_cli_overrides(args: list[str]) -> dict:
    """
    Parse ``--dotted.key value`` CLI tokens into a ``{dotted.key: value}`` dict.

    Each value is parsed as a Python literal, falling back to a plain string. A
    bare ``--key`` with no following value (at the end of the args or immediately
    before another ``--flag``) is treated as a boolean switch and set to ``True``.

    Parameters
    ----------
    args : list of str
        Extra CLI tokens, e.g. ``["--output.dir", "/tmp", "--implementation.debug_mode"]``.

    Returns
    -------
    dict
        Mapping of dotted key to its parsed value.
    """
    overrides = {}
    i = 0
    while i < len(args):
        arg = args[i]

        if not arg.startswith("--"):
            i += 1
            continue

        key = arg[2:]

        # A bare flag (end of args, or immediately followed by another --flag)
        # is a boolean switch and resolves to True, e.g. `--key`
        if i + 1 >= len(args) or args[i + 1].startswith("--"):
            Logger.debug(f"Setting {key} as True")
            overrides[key] = True
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

        overrides[key] = val
        i += 2

    return overrides


def parse_overrides(source: Union[click.Context, list[str], dict, None]) -> dict:
    """
    Normalize any override specification into a plain ``{dotted.key: value}`` dict.

    This is the single place that knows the different shapes overrides can arrive
    in, so the rest of the code only ever deals with a dict:

    - ``None`` -> ``{}`` (no overrides);
    - a ``click.Context`` -> its extra ``--dotted.key value`` args are parsed;
    - a ``list[str]`` of those same CLI tokens -> parsed directly;
    - a ``dict`` of dotted keys to already-typed values -> returned as-is.

    Only the CLI forms are literal-parsed; mapping values are used verbatim, so
    callers passing a dict provide real Python objects (``SNR=500``, not ``"500"``).

    Parameters
    ----------
    source : click.Context or list of str or dict or None
        The override specification in any supported shape.

    Returns
    -------
    dict
        Mapping of dotted key (e.g. ``"forward_model.instrument.SNR"``) to value.
    """
    if source is None:
        return {}
    if isinstance(source, dict):
        return source
    if isinstance(source, click.Context):
        return _parse_cli_overrides(source.args)
    return _parse_cli_overrides(source)


def override(box: dict, ctx: Union[click.Context, list[str], dict, None]) -> dict:
    """
    Patch the config in place with dotted ``key``/``value`` overrides.

    The overrides may be given in any shape accepted by :func:`parse_overrides`
    (``click.Context``, CLI arg list, or dict); normalization is delegated there,
    so this function only applies an already-canonical mapping. Keys are dotted
    (see :func:`~.common.set_dotted`), so ``"forward_model.instrument.SNR"``
    resolves into nested sections, creating any that are missing.

    A mapping overriding an existing section is deep-merged into it, keeping
    the section's other keys; any other value, including lists and ``None``,
    replaces the existing one outright.

    Parameters
    ----------
    box : dict
        The loaded config to override; mutated in place.
    ctx : click.Context or list of str or dict or None
        The overrides, in any shape accepted by :func:`parse_overrides`.

    Returns
    -------
    dict
        The same ``box``, updated with the overrides.
    """
    for key, val in parse_overrides(ctx).items():
        Logger.debug(f"Overriding {key} with {val!r}")
        set_dotted(box, key, val)

    return box
