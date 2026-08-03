from typing import Annotated, Union

from pydantic import BeforeValidator, Discriminator, Tag


def standardName(name: str) -> str:
    """
    Enables multiple names for each engine and standardizes them to a known name for
    the codebase. If the only acceptable name for an engine is the lowercased form, it
    is not needed to be added here.
    """
    name = name.lower()
    if name in {"6s", "sixs"}:
        return "sixs"
    return name


def selectByName(data: dict) -> str:
    """
    Discriminator function to select the config model for a given name
    """
    name = data.get("name")
    if isinstance(name, str):
        return standardName(name)
    return name


from .libradtran import LibRadTran
from .modtran import Modtran
from .sixs import SixS
from .srtmnet import sRTMnet

# Add new engines using their standard name
Engines = Annotated[
    Union[
        Annotated[LibRadTran, Tag("libradtran")],
        Annotated[Modtran, Tag("modtran")],
        Annotated[SixS, Tag("sixs")],
        Annotated[sRTMnet, Tag("srtmnet")],
    ],
    Discriminator(selectByName),
]
