"""
Shared file reading and nested-dict helpers for the config interpolation pipeline.

This is the leaf module of the package: every other stage imports from here,
and it imports from none of them.

Configs are plain nested ``dict`` / ``list`` trees. Nested values are addressed
with dotted keys (``"forward_model.instrument.SNR"``) through :func:`get_dotted`
and :func:`set_dotted`, and sections are layered with :func:`merge`.
"""

import copy
import json
from pathlib import Path
from typing import Any, Union

import yaml

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


def get_dotted(data: dict, key: str) -> Any:
    """
    Look up a value in a nested dict by dotted key.

    A literal key always wins over descending, and each dot is tried as a split
    point from left to right. This lets a key that itself contains dots be
    addressed: with ``{"env": {"srtmnet": "/p", "srtmnet.file": "f.h5"}}``,
    ``"env.srtmnet.file"`` resolves to ``"f.h5"``.

    Parameters
    ----------
    data : dict
        The mapping to search.
    key : str
        Dotted key, e.g. ``"forward_model.instrument.SNR"``.

    Returns
    -------
    Any
        The value at ``key``.

    Raises
    ------
    KeyError
        If ``key`` does not resolve to a value.
    """
    if key in data:
        return data[key]

    for i, char in enumerate(key):
        if char == "." and isinstance(data.get(key[:i]), dict):
            try:
                return get_dotted(data[key[:i]], key[i + 1 :])
            except KeyError:
                continue

    raise KeyError(key)


def set_dotted(data: dict, key: str, value: Any) -> None:
    """
    Set a value in a nested dict by dotted key, in place.

    Uses the same key resolution as :func:`get_dotted`. Missing or ``null``
    intermediate sections are created as empty dicts.

    Parameters
    ----------
    data : dict
        The mapping to update.
    key : str
        Dotted key, e.g. ``"forward_model.instrument.SNR"``.
    value : Any
        Value to assign.

    Raises
    ------
    TypeError
        If an intermediate section of ``key`` exists but is not a mapping.
    """
    if key in data or "." not in key:
        data[key] = value
        return

    for i, char in enumerate(key):
        if char == "." and isinstance(data.get(key[:i]), dict):
            set_dotted(data[key[:i]], key[i + 1 :], value)
            return

    head, rest = key.split(".", 1)
    if data.get(head) is not None:
        raise TypeError(
            f"Cannot set {key!r}: {head!r} is a {type(data[head]).__name__}, "
            "not a mapping"
        )

    data[head] = {}
    set_dotted(data[head], rest, value)


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
