"""
Prebuilt lookup table engine configuration.

This engine uses pre-computed atmospheric lookup tables without requiring
any radiative transfer code installation. Useful for rapid processing with
standard atmospheric conditions.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator

from . import standardName


class Prebuilt(BaseModel):
    """
    Prebuilt LUT engine configuration.

    This is the simplest engine option, using pre-computed lookup tables
    without requiring MODTRAN, 6S, or other RT codes. Suitable for standard
    atmospheric correction scenarios.

    Attributes
    ----------
    name : Literal["prebuilt"]
        Engine identifier. Must be "prebuilt".

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import Prebuilt
    >>> engine = Prebuilt(name="prebuilt")

    Notes
    -----
    Prebuilt LUTs are provided with ISOFIT for common sensor/atmosphere
    combinations. For custom atmospheric conditions or sensors, use a
    physics-based RT engine (MODTRAN, 6S, etc.) to generate custom LUTs.

    See Also
    --------
    Modtran : MODTRAN 6.0 RT engine
    SixS : 6S RT engine
    sRTMnet : Machine learning RT emulator
    """

    name: Annotated[Literal["prebuilt"], BeforeValidator(standardName)]
