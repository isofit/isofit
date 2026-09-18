"""
6S (Second Simulation of the Satellite Signal in the Solar Spectrum) engine configuration.

6S is an open-source atmospheric RT code widely used in remote sensing.
This module configures 6S for ISOFIT atmospheric correction.
"""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)

from isofit.configs.utils.validators import PathExists
from isofit.data import env

from . import standardName


class SixS(BaseModel):
    """
    6S engine configuration.

    6S (Second Simulation of the Satellite Signal in the Solar Spectrum) is
    an open-source RT code for solar-reflected wavelengths (visible to SWIR).

    Attributes
    ----------
    name : Literal["sixs"]
        Engine identifier. Must be "sixs" or "6s".
    base_dir : PathExists
        Base path to 6S installation directory. If not provided, falls back
        to environment configuration.
    day : int
        Day of year (1-366) for solar geometry calculations.
    month : int
        Month (1-12) for solar geometry calculations.
    elev : float
        Surface elevation in kilometers (positive).
    alt : float
        Sensor altitude in kilometers (positive).
    solzen : float
        Solar zenith angle in degrees (0-180).
    solaz : float
        Solar azimuth angle in degrees.
    viewzen : float
        View zenith angle in degrees.
    viewaz : float
        View azimuth angle in degrees.
    obs : PathExists
        Path to 6S observation file with geometry parameters.
    earth_sun_distance_file : PathExists
        Path to Earth-Sun distance file for irradiance corrections.
    irradiance_file : PathExists
        Path to solar irradiance file.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import SixS
    >>> sixs = SixS(
    ...     name="6s",
    ...     base_dir="/opt/6S",
    ...     day=172,
    ...     month=6,
    ...     elev=0.5,
    ...     alt=705.0,
    ...     solzen=30.0,
    ...     solaz=180.0,
    ...     viewzen=0.0,
    ...     viewaz=0.0,
    ...     obs="geometry.txt",
    ...     earth_sun_distance_file="earth_sun_dist.txt",
    ...     irradiance_file="solar_irrad.txt"
    ... )

    Notes
    -----
    6S is freely available and commonly used for satellite missions.
    Download from: http://6s.ltdri.org/

    6S is generally faster than MODTRAN but has fewer atmospheric profile
    options and does not support thermal infrared wavelengths.

    See Also
    --------
    Modtran : Commercial RT engine with more capabilities
    LibRadTran : Alternative open-source RT engine
    """

    name: Annotated[Literal["sixs"], BeforeValidator(standardName)]

    base_dir: PathExists = Field(
        default=None,
        description="Base path to engine directory",
    )

    day: int = Field(ge=1, le=366, description="Day Parameter", examples=[1, 365])

    month: int = Field(ge=1, le=12, description="Month parameter", examples=[1, 12])

    elev: PositiveFloat = Field(
        description="Elevation parameter",
    )

    alt: PositiveFloat = Field(
        description="Altitude parameter",
    )

    solzen: float = Field(
        ge=0,
        le=180,
        description="Solar zenith parameter",
    )

    solaz: float = Field(
        description="Solar azimuth parameter",
    )

    viewzen: float = Field(
        description="View zenith parameter",
    )

    viewaz: float = Field(
        description="View azimuth parameter",
    )

    obs: PathExists = Field(
        description="6S observation file",
    )

    earth_sun_distance_file: PathExists = Field(
        description="Earth-Sun distance file",
    )

    irradiance_file: PathExists = Field(
        description="Irradiance file",
    )

    @model_validator(mode="before")
    @classmethod
    def env_fallback(cls, data: dict, info: dict) -> dict:
        """
        Handle base_dir environment fallback.

        If base_dir is not defined in configuration, falls back to environment
        settings. If base_dir is defined, updates the current environment to
        use the new base path (without saving to disk).

        Parameters
        ----------
        data : dict
            Raw configuration data before validation.
        info : dict
            Validation context information from Pydantic.

        Returns
        -------
        dict
            Configuration data with base_dir set from config or environment.
        """
        if base := data.get("base_dir"):
            env.changePath("sixs", base)
        data["base_dir"] = env.sixs

        return data
