"""
Shared file reading and nested-dict helpers for the config interpolation pipeline.

This is the leaf module of the package: every other stage imports from here,
and it imports from none of them.

Configs are plain nested ``dict`` / ``list`` trees. Nested values are addressed
with dotted keys (``"forward_model.instrument.SNR"``), with ``[i]`` for list
items (``"implementation.inversion.windows[0]"``), through :func:`get_dotted`
and :func:`set_dotted`, and sections are layered with :func:`merge`.
"""

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator, Union

import yaml

Logger = logging.getLogger(__name__)

# Supported config file extensions, mapped to the parser for that format.
Loaders = {
    ".yml": yaml.safe_load,
    ".yaml": yaml.safe_load,
    ".json": json.load,
}


def read(config: Union[str, Path]) -> dict:
    """
    Load a config file into a dict, dispatching on its extension.

    Both YAML (``.yml`` / ``.yaml``) and JSON (``.json``) are supported. An
    empty file loads as an empty dict.

    Parameters
    ----------
    config : str or pathlib.Path
        File path to the config to read.

    Returns
    -------
    dict
        The loaded configuration.

    Raises
    ------
    ValueError
        If the file's extension is not a supported config format, or the file's
        top level is not a mapping.
    """
    suffix = Path(config).suffix.lower()
    if suffix not in Loaders:
        supported = ", ".join(sorted(Loaders))
        raise ValueError(
            f"Unsupported config file extension {suffix!r} for {str(config)!r}; "
            f"expected one of: {supported}"
        )

    with open(config) as file:
        data = Loaders[suffix](file)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a mapping at the top level of {str(config)!r}, "
            f"got {type(data).__name__}"
        )
    return data


def split_points(key: str) -> Iterator[tuple[str, str, type]]:
    """
    Yield every way to split ``key`` into a head and the rest, left to right.

    A ``.`` separates two mapping keys and is dropped. A ``[`` starts a list
    index and is kept on the rest.

    Parameters
    ----------
    key : str
        Dotted key, e.g. ``"a.b[0].c"``.

    Yields
    ------
    head : str
        The key before the split point.
    rest : str
        The key after the split point.
    kind : type
        The container ``head`` must hold for ``rest`` to apply to it: ``dict``
        for a ``.`` split, ``list`` for a ``[`` split.
    """
    for i, char in enumerate(key):
        if char == ".":
            yield key[:i], key[i + 1 :], dict
        elif char == "[" and i:
            yield key[:i], key[i:], list


def split_index(data: list, key: str) -> tuple[int, str]:
    """
    Split a leading ``[i]`` off ``key``, checking ``i`` is in range of ``data``.

    Parameters
    ----------
    data : list
        The list being indexed.
    key : str
        Key starting with a list index, e.g. ``"[0].name"``.

    Returns
    -------
    i : int
        The index. Negative indices count from the end.
    rest : str
        The remainder of the key, with any leading ``.`` dropped.

    Raises
    ------
    KeyError
        If ``key`` does not start with ``[i]``, ``i`` is out of range, or the
        index is not followed by ``.``, ``[``, or the end of the key.
    """
    # A leading ``[i]`` list index on a key, e.g. the ``[1]`` of ``"[1].name"``
    match = re.match(r"\[(-?\d+)\]", key)
    if not match or not -len(data) <= int(match[1]) < len(data):
        raise KeyError(key)

    rest = key[match.end() :]
    if rest.startswith("."):
        rest = rest[1:]
    elif rest and not rest.startswith("["):
        raise KeyError(key)

    return int(match[1]), rest


def get_dotted(data: Union[dict, list], key: str) -> Any:
    """
    Look up a value in a nested dict/list tree by dotted key.

    Mapping keys are separated by ``.`` and list items are indexed with
    ``[i]``, so ``"a.b[1].c"`` is ``data["a"]["b"][1]["c"]``. Negative indices
    count from the end.

    A literal key always wins over descending, and each separator is tried as a
    split point from left to right. This lets a key that itself contains dots be
    addressed: with ``{"env": {"srtmnet": "/p", "srtmnet.file": "f.h5"}}``,
    ``"env.srtmnet.file"`` resolves to ``"f.h5"``.

    Parameters
    ----------
    data : dict or list
        The tree to search.
    key : str
        Dotted key, e.g. ``"forward_model.instrument.SNR"`` or
        ``"implementation.inversion.windows[0]"``. An empty key returns
        ``data`` itself.

    Returns
    -------
    Any
        The value at ``key``.

    Raises
    ------
    KeyError
        If ``key`` does not resolve to a value, including when it indexes past
        the end of a list or descends into a scalar.

    Examples
    --------
    >>> data = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    >>> get_dotted(data, "a.b[1].c")
    2
    >>> get_dotted(data, "a.b[-1]")
    {'c': 2}
    """
    if not key:
        return data

    if isinstance(data, list):
        i, rest = split_index(data, key)
        return get_dotted(data[i], rest)

    if not isinstance(data, dict):
        raise KeyError(key)

    if key in data:
        return data[key]

    for head, rest, kind in split_points(key):
        if isinstance(data.get(head), kind):
            try:
                return get_dotted(data[head], rest)
            except KeyError:
                continue

    raise KeyError(key)


def set_dotted(data: Union[dict, list], key: str, value: Any) -> None:
    """
    Set a value in a nested dict/list tree by dotted key, in place.

    Uses the same key resolution as :func:`get_dotted`, including ``[i]`` list
    indexing. Missing or ``null`` intermediate sections are created as empty
    dicts. Lists are never created or extended, so an indexed item must
    already exist.

    A mapping assigned onto an existing mapping is deep-merged into it (see
    :func:`merge`), so other keys of that section are kept. Any other value,
    including lists and ``None``, replaces the existing one outright. The value
    is deep-copied, so ``data`` never aliases it.

    Parameters
    ----------
    data : dict or list
        The tree to update.
    key : str
        Dotted key, e.g. ``"forward_model.instrument.SNR"`` or
        ``"implementation.inversion.windows[0]"``.
    value : Any
        Value to assign.

    Raises
    ------
    KeyError
        If a list index in ``key`` is malformed or out of range.
    TypeError
        If an intermediate section of ``key`` is not the container the key
        expects: a mapping before ``.``, a list before ``[i]``.

    Examples
    --------
    >>> data = {"a": {"b": [1, 2], "c": {"x": 1}}, "n": None}
    >>> set_dotted(data, "a.b[-1]", 3)
    >>> set_dotted(data, "a.c", {"y": 2})
    >>> set_dotted(data, "n.k", 4)
    >>> data
    {'a': {'b': [1, 3], 'c': {'x': 1, 'y': 2}}, 'n': {'k': 4}}
    """
    if isinstance(data, list):
        i, rest = split_index(data, key)
        if rest:
            set_within(data, i, rest, value, key)
        else:
            assign(data, i, value)
        return

    first = next(split_points(key), None)
    if key in data or first is None:
        assign(data, key, value)
        return

    for head, rest, kind in split_points(key):
        if isinstance(data.get(head), kind):
            set_dotted(data[head], rest, value)
            return

    # No existing section matches, so descend at the first split, creating it
    head, rest, _ = first
    set_within(data, head, rest, value, key)


def assign(parent: Union[dict, list], slot: Union[str, int], value: Any) -> None:
    """
    Set ``parent[slot]`` to a copy of ``value``, deep-merging when both it and
    the existing value are mappings.

    Parameters
    ----------
    parent : dict or list
        The container to update.
    slot : str or int
        The key or index in ``parent`` to set.
    value : Any
        Value to assign.
    """
    current = parent.get(slot) if isinstance(parent, dict) else parent[slot]
    if isinstance(value, dict) and isinstance(current, dict):
        merge(current, value)
    else:
        parent[slot] = copy.deepcopy(value)


def set_within(
    parent: Union[dict, list], slot: Union[str, int], rest: str, value: Any, key: str
) -> None:
    """
    Set ``rest`` within ``parent[slot]``, creating a mapping there if it is
    missing or ``None`` and ``rest`` names a mapping key.

    Parameters
    ----------
    parent : dict or list
        The container holding the section to descend into.
    slot : str or int
        The key or index of that section in ``parent``.
    rest : str
        The remainder of the key, set within the section.
    value : Any
        Value to assign.
    key : str
        The key being set, for error messages.

    Raises
    ------
    TypeError
        If the section is not a mapping (``rest`` is a key) or not a list
        (``rest`` starts with ``[i]``).
    """
    kind = list if rest.startswith("[") else dict
    child = parent.get(slot) if isinstance(parent, dict) else parent[slot]

    if child is None and kind is dict:
        child = parent[slot] = {}

    if not isinstance(child, kind):
        expected = "list" if kind is list else "mapping"
        raise TypeError(
            f"Cannot set {key!r}: {slot!r} is a {type(child).__name__}, not a {expected}"
        )

    set_dotted(child, rest, value)


def merge(dst: dict, src: dict) -> dict:
    """
    Deep-merge ``src`` into ``dst`` in place; ``src`` wins on conflict.

    Nested mappings are merged key by key. Every other value, including lists
    and an explicit ``None``, replaces the value in ``dst`` outright. Values
    taken from ``src`` are deep-copied, so later mutation of ``dst`` never
    reaches back into ``src``.

    Parameters
    ----------
    dst : dict
        The mapping to update.
    src : dict
        The mapping to merge on top.

    Returns
    -------
    dict
        The same ``dst``, updated.
    """
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            merge(dst[key], val)
        else:
            dst[key] = copy.deepcopy(val)

    return dst
