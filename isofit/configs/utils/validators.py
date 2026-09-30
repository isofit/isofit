"""
Shared utility validation functions for pyconf.

This module provides Pydantic validators and custom type annotations for
validating file paths and other configuration values used throughout the
isofit.configs module.
"""

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

from isofit.data import env


def exists(path: Path) -> Path:
    """
    Validate that a path exists on the filesystem.

    This validator is used with Pydantic to ensure that file paths specified
    in configuration files actually exist before processing begins.

    Parameters
    ----------
    path : Path
        Path object to validate.

    Returns
    -------
    Path
        The same path object if validation succeeds.

    Raises
    ------
    ValueError
        If the path does not exist on the filesystem.

    Examples
    --------
    >>> from pathlib import Path
    >>> exists(Path("/tmp"))  # Returns Path("/tmp") if it exists
    PosixPath('/tmp')
    >>> exists(Path("/nonexistent"))  # Raises ValueError
    Traceback (most recent call last):
        ...
    ValueError: path does not exist

    Notes
    -----
    This function is typically not called directly but used as a Pydantic
    validator through the PathExists type annotation.
    """
    if not path.exists():
        raise ValueError("path does not exist")
    return path


PathExists = Annotated[Path, AfterValidator(exists)]
"""
Type annotation for Path fields that must exist on the filesystem.

This is a Pydantic-compatible type that combines pathlib.Path with automatic
validation that the path exists. Use this for configuration fields that specify
input files that must be present.

Examples
--------
>>> from pydantic import BaseModel, Field
>>> class MyConfig(BaseModel):
...     input_file: PathExists = Field(description="Path to input file")
>>> config = MyConfig(input_file="/tmp")  # Validates /tmp exists
>>> config = MyConfig(input_file="/nonexistent")  # Raises ValidationError
"""
