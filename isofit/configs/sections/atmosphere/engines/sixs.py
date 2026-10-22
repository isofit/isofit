"""
6S (Second Simulation of the Satellite Signal in the Solar Spectrum) engine configuration.

6S is an open-source atmospheric RT code widely used in remote sensing.
This module configures 6S for ISOFIT atmospheric correction.
"""

from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
)

from . import standardName


class SixSConfig(BaseModel):
    """
    6S engine configuration.

    6S (Second Simulation of the Satellite Signal in the Solar Spectrum) is
    an open-source RT code for solar-reflected wavelengths (visible to SWIR).

    Examples
    --------
    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "atmosphere": {
    ...             "engine": {
    ...                 "name": "6s",
    ...                 "day": 172,
    ...                 "month": 6,
    ...                 "elev": 0.5,
    ...                 "alt": 705.0,
    ...                 "solzen": 30.0,
    ...                 "solaz": 180.0,
    ...                 "viewzen": 0.0,
    ...                 "viewaz": 0.0,
    ...             }
    ...         }
    ...     }
    ... })
    >>> sixs = config.forward_model.atmosphere.engine
    >>> sixs.name
    'sixs'

    Notes
    -----
    6S is freely available and commonly used for satellite missions.
    Download from: http://6s.ltdri.org/

    6S is generally faster than MODTRAN but has fewer atmospheric profile
    options and does not support thermal infrared wavelengths.

    The solar irradiance file is not part of this engine config; it is set via
    ``forward_model.atmosphere.irradiance_file`` (see :class:`AtmosphereConfig`),
    which 6S requires.

    See Also
    --------
    ModtranConfig : Commercial RT engine with more capabilities
    LibRadTranConfig : Alternative open-source RT engine
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["sixs"], BeforeValidator(standardName)] = Field(
        description=(
            'Engine identifier. Accepts "sixs" or "6s" (case-insensitive), '
            'normalized to "sixs"'
        )
    )

    base_dir: Optional[DirectoryPath] = Field(
        description=(
            "Base path to the 6S installation directory. The base configs default "
            "this to the ISOFIT ini ``sixs`` path"
        ),
    )

    # The solar/observer geometry parameters below are populated dynamically by
    # ``build_sixs_config`` (from a MODTRAN template) when 6S is run as the
    # sRTMnet surrogate, so they are nullable rather than carrying real values.
    day: Optional[int] = Field(
        ge=1,
        le=366,
        description="Day of year (1-366) for solar geometry calculations",
        examples=[1, 365],
    )

    month: Optional[int] = Field(
        ge=1,
        le=12,
        description="Month (1-12) for solar geometry calculations",
        examples=[1, 12],
    )

    elev: Optional[float] = Field(
        description="Surface elevation in kilometers (positive)",
    )

    alt: Optional[float] = Field(
        description="Sensor altitude in kilometers (positive)",
    )

    solzen: Optional[float] = Field(
        ge=0,
        le=180,
        description="Solar zenith angle in degrees (0-180)",
    )

    solaz: Optional[float] = Field(
        description="Solar azimuth angle in degrees",
    )

    viewzen: Optional[float] = Field(
        description="View zenith angle in degrees",
    )

    viewaz: Optional[float] = Field(
        description="View azimuth angle in degrees",
    )

    wlinf: Optional[float] = Field(
        description="Shortest wavelength (microns) to run the simulation for",
    )

    wlsup: Optional[float] = Field(
        description="Longest wavelength (microns) to run the simulation for",
    )

    template_file: Optional[FilePath] = Field(
        description="MODTRAN template file used to populate 6S geometry",
    )

    aerosol_model_file: Optional[FilePath] = Field(
        description="Aerosol model file",
    )

    aerosol_template_file: Optional[FilePath] = Field(
        description="Aerosol template file",
    )

    obs: Optional[FilePath] = Field(
        description="Path to the 6S observation file with geometry parameters",
    )

    earth_sun_distance_file: Optional[FilePath] = Field(
        description="Path to the Earth-Sun distance file for irradiance corrections",
    )
