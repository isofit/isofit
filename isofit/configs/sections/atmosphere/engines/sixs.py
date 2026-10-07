"""
6S (Second Simulation of the Satellite Signal in the Solar Spectrum) engine configuration.

6S is an open-source atmospheric RT code widely used in remote sensing.
This module configures 6S for ISOFIT atmospheric correction.
"""

from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)

from isofit.configs.utils.validators import PathExists
from isofit.data import env

from . import standardName


class SixSConfig(BaseModel):
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
    >>> from isofit.configs.sections.atmosphere.engines import SixSConfig
    >>> sixs = SixSConfig(
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
    ModtranConfig : Commercial RT engine with more capabilities
    LibRadTranConfig : Alternative open-source RT engine
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["sixs"], BeforeValidator(standardName)]

    base_dir: Optional[PathExists] = Field(
        description="Base path to engine directory",
    )

    # The solar/observer geometry parameters below are populated dynamically by
    # ``build_sixs_config`` (from a MODTRAN template) when 6S is run as the
    # sRTMnet surrogate, so they are nullable rather than carrying real values.
    day: Optional[int] = Field(
        ge=1, le=366, description="Day Parameter", examples=[1, 365]
    )

    month: Optional[int] = Field(
        ge=1, le=12, description="Month parameter", examples=[1, 12]
    )

    elev: Optional[float] = Field(
        description="Elevation parameter",
    )

    alt: Optional[float] = Field(
        description="Altitude parameter",
    )

    solzen: Optional[float] = Field(
        ge=0,
        le=180,
        description="Solar zenith parameter",
    )

    solaz: Optional[float] = Field(
        description="Solar azimuth parameter",
    )

    viewzen: Optional[float] = Field(
        description="View zenith parameter",
    )

    viewaz: Optional[float] = Field(
        description="View azimuth parameter",
    )

    wlinf: Optional[float] = Field(
        description="Shortest wavelength (microns) to run the simulation for",
    )

    wlsup: Optional[float] = Field(
        description="Longest wavelength (microns) to run the simulation for",
    )

    template_file: Optional[PathExists] = Field(
        description="MODTRAN template file used to populate 6S geometry",
    )

    aerosol_model_file: Optional[PathExists] = Field(
        description="Aerosol model file",
    )

    aerosol_template_file: Optional[PathExists] = Field(
        description="Aerosol template file",
    )

    obs: Optional[PathExists] = Field(
        description="6S observation file",
    )

    earth_sun_distance_file: Optional[PathExists] = Field(
        description="Earth-Sun distance file",
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
