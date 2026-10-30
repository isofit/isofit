"""
Pipeline entry point: read a config file and run every stage over it.

``read -> patch (^^) -> override -> interpolate (${...})``
"""

import copy
from pathlib import Path
from typing import Optional

import click

from .common import read
from .inherit import patch
from .overrides import override
from .resolve import interpolate


def load(
    config: str,
    section: Optional[str] = None,
    ctx: Optional[click.Context] = None,
    interp: bool = True,
) -> dict:
    """
    Loads a config from a yaml or json file

    Parameters
    ----------
    config : str
        File path to the config (.yml, .yaml, or .json)
    section : str
        Returns only this subsection of the config
    ctx : click.Context or list of str or dict, default=None
        Overrides, in any shape accepted by :func:`~.overrides.parse_overrides`
    interp : bool, default=True
        Calls interpolate on the config before returning

    Returns
    -------
    config : dict
        Loaded configuration

    Raises
    ------
    KeyError
        If ``section`` is not a top-level section of the file.
    """
    box = full = read(config)

    if section:
        if section not in full:
            available = ", ".join(full)
            raise KeyError(
                f"Section {section!r} not found in {str(config)!r}; "
                f"available sections: {available}"
            )
        # Copied so patch() popping `^^` leaves `full` intact; same-file refs
        # back to this section must still see its `^^` for cycle detection
        box = copy.deepcopy(full[section])

    if "^^" in box:
        path = Path(config).resolve()
        box = patch(box, full, base_dir=path.parent, cache={path: full}, file=path)

    if ctx:
        box = override(box, ctx)

    if interp:
        box = interpolate(box)

    return box
