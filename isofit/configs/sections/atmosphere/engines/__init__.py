"""
Atmospheric radiative transfer engine configurations.

This module provides Pydantic models for various RT engines supported by ISOFIT,
including MODTRAN, 6S, LibRadTran, sRTMnet, and prebuilt LUTs. It uses discriminated
unions to handle different engine types with a common interface.
"""

from typing import Annotated, Union

from pydantic import BeforeValidator, Discriminator, Tag


def standardName(name: str) -> str:
    """
    Standardize engine names to canonical form.

    Enables multiple name aliases for each engine and normalizes them to a
    standard name used internally in the codebase. If the only acceptable
    name for an engine is its lowercased form, it does not need special handling.

    Parameters
    ----------
    name : str
        Engine name to standardize (case-insensitive).

    Returns
    -------
    str
        Standardized engine name.

    Examples
    --------
    >>> standardName("6S")
    'sixs'
    >>> standardName("SIXS")
    'sixs'
    >>> standardName("kf")
    'kernelflows'
    >>> standardName("MODTRAN")
    'modtran'

    Notes
    -----
    Supported aliases:
    - "6s", "sixs" -> "sixs"
    - "kf", "kernelflows" -> "kernelflows"
    - All other names are lowercased
    """
    name = name.lower()
    if name in {"6s", "sixs"}:
        return "sixs"
    elif name in {"kf", "kernelflows"}:
        return "kernelflows"
    return name


def selectByName(data: dict) -> str:
    """
    Discriminator function to select the config model for a given engine name.

    Used by Pydantic's discriminated union to route configuration dictionaries
    to the correct engine model class based on the 'name' field.

    Parameters
    ----------
    data : dict
        Configuration dictionary containing 'name' field.

    Returns
    -------
    str
        Standardized engine name for discriminator matching.

    Notes
    -----
    This function is called automatically by Pydantic during validation
    and should not be invoked directly by users.
    """
    name = data.get("name")
    if isinstance(name, str):
        return standardName(name)
    return name


from .libradtran import LibRadTranConfig
from .modtran import ModtranConfig
from .prebuilt import PrebuiltConfig
from .sixs import SixSConfig
from .srtmnet import sRTMnetConfig

# Add new engines using their standard name
Engines = Annotated[
    Union[
        Annotated[LibRadTranConfig, Tag("libradtran")],
        Annotated[ModtranConfig, Tag("modtran")],
        Annotated[PrebuiltConfig, Tag("prebuilt")],
        Annotated[SixSConfig, Tag("sixs")],
        Annotated[sRTMnetConfig, Tag("srtmnet")],
    ],
    Discriminator(selectByName),
]
