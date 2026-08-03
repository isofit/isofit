"""
Shared utility validation functions
"""

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BeforeValidator

from isofit.data import env


def exists(path: Path) -> Path:
    """
    Validate a path exists
    """
    if not path.exists():
        raise ValueError("path does not exist")
    return path


Exists = Annotated[Path, AfterValidator(exists)]


def env_fallback(path: Path | None) -> Path:
    """ """
    if not path.exists():
        ...
