"""
MODTRAN radiative transfer engine configuration.

MODTRAN is a widely-used atmospheric RT code for visible through thermal infrared
wavelengths. This module configures MODTRAN 6.0 for ISOFIT atmospheric correction.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from isofit.configs.utils.validators import PathExists
from isofit.data import env

from . import standardName


class ModtranConfig(BaseModel):
    """
    MODTRAN 6.0 engine configuration.

    MODTRAN (MODerate resolution atmospheric TRANsmission) is a physics-based
    RT code supporting VNIR to TIR wavelengths with comprehensive atmospheric
    absorption and scattering.

    Attributes
    ----------
    name : Literal["modtran"]
        Engine identifier. Must be "modtran".
    base_dir : PathExists
        Base path to MODTRAN installation directory. If not provided,
        falls back to environment configuration.
    multipart_transmittance : bool
        Apply triple-run diffuse & direct transmittance estimation for
        improved accuracy. Increases runtime 3x. Default is False.
    aerosol_template_file : PathExists
        Path to aerosol template file defining aerosol optical properties
        and vertical profiles.
    aerosol_model_file : PathExists
        Path to aerosol model file with wavelength-dependent properties.
    template_file : PathExists
        Path to MODTRAN input template file (tape5 or JSON format) with
        base atmospheric configuration.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import ModtranConfig
    >>> modtran = ModtranConfig(
    ...     name="modtran",
    ...     base_dir="/opt/modtran6.0",
    ...     aerosol_template_file="aerosol_template.txt",
    ...     aerosol_model_file="aerosol_model.txt",
    ...     template_file="modtran_template.json",
    ...     multipart_transmittance=True
    ... )

    Notes
    -----
    MODTRAN requires a valid license and installation. See MODTRAN
    documentation for installation and licensing:
    http://modtran.spectral.com/

    The template file defines the base atmospheric profile, viewing geometry
    format, and RT solver settings. ISOFIT modifies this template for each
    LUT point.

    See Also
    --------
    SixSConfig : Alternative open-source RT engine
    LibRadTranConfig : Alternative open-source RT engine
    """

    name: Annotated[Literal["modtran"], BeforeValidator(standardName)]

    base_dir: PathExists = Field(
        default=None,
        description="Base path to engine directory",
    )

    multipart_transmittance: bool = Field(
        default=False,
        description="Apply triple-run diffuse & direct transmittance estimation",
    )

    aerosol_template_file: PathExists = Field(
        description="Aerosol template file",
    )

    aerosol_model_file: PathExists = Field(
        description="Aerosol model file",
    )

    template_file: PathExists = Field(
        description="",
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

        Notes
        -----
        This allows MODTRAN installation path to be configured once in
        environment settings and reused across multiple configurations.
        """
        if base := data.get("base_dir"):
            env.changePath("modtran", base)
        data["base_dir"] = env.modtran

        return data
