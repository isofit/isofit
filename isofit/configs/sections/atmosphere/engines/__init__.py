"""
Atmospheric radiative transfer engine configurations.

This module provides Pydantic models for various RT engines supported by ISOFIT,
including MODTRAN, 6S, LibRadTran, sRTMnet, and prebuilt LUTs. It uses discriminated
unions to handle different engine types with a common interface.
"""

from typing import Annotated, Union

from pydantic import AliasChoices, BeforeValidator, Discriminator, Tag

# Validation alias shared by every engine's `name` field; `engine_name` is the legacy key
NameAlias = AliasChoices("name", "engine_name")


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
    and should not be invoked directly by users. Discriminators run before
    aliases are resolved, so the legacy ``engine_name`` key is checked here
    explicitly.
    """
    name = data.get("name", data.get("engine_name"))
    if isinstance(name, str):
        return standardName(name)
    return name


def inputKey(model, data: dict, field: str) -> str:
    """
    Find the key in raw input data that supplies a field.

    ``mode="before"`` validators run before aliases are resolved, so they must
    read and write a field under whichever of its validation aliases was given.

    Parameters
    ----------
    model : type[pydantic.BaseModel]
        Model class that defines ``field``.
    data : dict
        Raw configuration data before validation.
    field : str
        Field name to look up.

    Returns
    -------
    str
        The first of the field's aliases present in ``data``, otherwise the
        field name itself.
    """
    alias = model.model_fields[field].validation_alias
    choices = alias.choices if isinstance(alias, AliasChoices) else [field]
    return next((key for key in choices if key in data), field)


from .kernelflows import KernelFlowsConfig
from .libradtran import LibRadTranConfig
from .modtran import ModtranConfig
from .prebuilt import PrebuiltConfig
from .sixs import SixSConfig
from .srtmnet import sRTMnetConfig

# Add new engines using their standard name
Engines = Annotated[
    Union[
        Annotated[KernelFlowsConfig, Tag("kernelflows")],
        Annotated[LibRadTranConfig, Tag("libradtran")],
        Annotated[ModtranConfig, Tag("modtran")],
        Annotated[PrebuiltConfig, Tag("prebuilt")],
        Annotated[SixSConfig, Tag("sixs")],
        Annotated[sRTMnetConfig, Tag("srtmnet")],
    ],
    Discriminator(selectByName),
]
