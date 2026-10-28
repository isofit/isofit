"""
Stage 3 of the pipeline: ``${...}`` reference interpolation.

:func:`interpolate` walks the config tree and hands each string to
:func:`interp`, which substitutes ``${key}`` (absolute) and ``${.key}``
(relative) references.
"""

import ast
import logging
import re
from typing import Any, Optional, Union

from .common import locate

Logger = logging.getLogger(__name__)
Interp = re.compile(r"\${([^}]+)}")


def interp(val: str, rel: dict, full: dict, _seen: Optional[set] = None) -> Any:
    """
    Interpolate ``${...}`` references in a single string against the config.

    Format: ``${.key}`` or ``${key}``. A leading dot makes the key relative to the
    subsection the original value lives in (``rel``); otherwise it resolves from the
    top of the config (``full``). After substitution the result is parsed as a Python
    literal when possible, so a fully-substituted string can become e.g. an int/list.

    Referenced values are resolved depth-first in their own context before being
    substituted: a referenced string has its relative refs resolved against its
    own section, and a referenced list or mapping is fully interpolated. Either
    way the resolved value is written back into the config, so each value is
    resolved once and the result does not depend on key order.

    Parameters
    ----------
    val : str
        The value to interpolate.
    rel : dict
        The subsection used to resolve relative (``${.key}``) references.
    full : dict
        The full config used to resolve absolute (``${key}``) references.
    _seen : set, optional
        Internal set of the values currently being resolved on this recursion
        chain, used to detect circular references. Should not be passed by
        callers.

    Returns
    -------
    Any
        The interpolated value; a literal (int/list/...) when it parses as one,
        otherwise the substituted string (or ``val`` unchanged if it had no refs).
    """
    new = val
    if matches := Interp.findall(val):
        if _seen is None:
            _seen = set()

        for key in matches:
            match = "${" + key + "}"

            if key.startswith("."):
                Logger.debug("Using relative pathing")
                ref = rel
                lookup = key[1:]
            else:
                Logger.debug("Using full pathing")
                ref = full
                lookup = key

            try:
                # Find where the substitute value lives
                path = locate(ref, lookup)
            except KeyError:
                msg = f"Interpolation reference {match!r} not found in config"
                Logger.error(msg)
                raise KeyError(msg) from None

            sub = path[-1][0][path[-1][1]] if path else ref

            # Containers are identified by themselves, strings by their slot
            if isinstance(sub, (dict, list)):
                ident = id(sub)
            else:
                ident = (id(path[-1][0]), path[-1][1])

            if ident in _seen:
                msg = f"Circular interpolation reference detected for {match!r}"
                Logger.error(msg)
                raise ValueError(msg)

            # The substitute's own relative refs resolve against the nearest
            # mapping holding it, as they would during the walk
            sub_rel = next(
                (box for box, _ in reversed(path) if isinstance(box, dict)), ref
            )

            # Resolve the substitute in its own context first and write it back
            if isinstance(sub, (dict, list)):
                interpolate(sub, full, sub_rel, _seen)
            elif isinstance(sub, str) and "${" in sub:
                parent, slot = path[-1]
                sub = parent[slot] = interp(sub, sub_rel, full, _seen | {ident})

            new = new.replace(match, str(sub))
            Logger.debug(f"Replaced {match!r} with {sub!r}")

        try:
            new = ast.literal_eval(new)
        except (ValueError, SyntaxError):
            pass

        if new is not val:
            Logger.debug(f"New value: {new!r}")

    return new


def interpolate(
    box: Union[dict, list],
    full: Optional[dict] = None,
    rel: Optional[dict] = None,
    _seen: Optional[set] = None,
) -> Union[dict, list]:
    """
    Recursively interpolate every ``${...}`` reference in a config tree in place.

    Walks ``box`` (a dict or list), replacing each string value via :func:`interp`.
    ``full`` is the full config used for absolute references; ``rel`` is the current
    subsection used for relative ones. Both default to ``box`` itself on the first
    call and are threaded down as the walk descends into subsections.

    :func:`interp` resolves each referenced value depth-first and writes it
    back, so the result is the same whatever order the keys are walked in.

    Parameters
    ----------
    box : dict or list
        The config node to interpolate; mutated in place.
    full : dict, optional
        Full config for absolute references. Defaults to ``box`` on the first call.
    rel : dict, optional
        Subsection for relative references. Defaults to ``full``.
    _seen : set, optional
        Internal set of the values currently being resolved, used by
        :func:`interp` to detect circular references. Should not be passed by
        callers.

    Returns
    -------
    dict or list
        The same ``box``, with all ``${...}`` references interpolated.
    """
    if full is None:
        full = box

    if rel is None:
        rel = full

    # A reference to any container being walked, from inside it, is circular
    _seen = (_seen or set()) | {id(box)}

    if isinstance(box, list):
        items = enumerate(box)
    else:
        rel = box
        items = box.items()

    for key, val in items:
        if isinstance(val, (dict, list)):
            interpolate(val, full, rel, _seen)
        elif isinstance(val, str):
            box[key] = interp(val, rel, full, _seen | {(id(box), key)})

    return box
