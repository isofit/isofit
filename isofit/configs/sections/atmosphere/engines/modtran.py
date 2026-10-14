"""
MODTRAN radiative transfer engine configuration.

MODTRAN is a widely-used atmospheric RT code for visible through thermal infrared
wavelengths. This module configures MODTRAN 6.0 for ISOFIT atmospheric correction.
"""

from typing import Annotated, Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
)

from . import NameAlias, standardName


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
    base_dir : DirectoryPath
        Base path to MODTRAN installation directory. The base configs default
        this to the ISOFIT ini ``modtran`` path. Accepts the legacy key
        ``engine_base_dir``.
    multipart_transmittance : bool
        Apply triple-run diffuse & direct transmittance estimation for
        improved accuracy. Increases runtime 3x. Default is False.
    aerosol_template_file : FilePath
        Path to aerosol template file defining aerosol optical properties
        and vertical profiles.
    aerosol_model_file : FilePath
        Path to aerosol model file with wavelength-dependent properties.
    template_file : FilePath
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

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["modtran"], BeforeValidator(standardName)] = Field(
        validation_alias=NameAlias,
    )

    base_dir: Optional[DirectoryPath] = Field(
        validation_alias=AliasChoices("base_dir", "engine_base_dir"),
        description="Base path to engine directory (legacy alias: engine_base_dir)",
    )

    multipart_transmittance: bool = Field(
        description="Apply triple-run diffuse & direct transmittance estimation",
    )

    aerosol_template_file: Optional[FilePath] = Field(
        description="Aerosol template file",
    )

    aerosol_model_file: Optional[FilePath] = Field(
        description="Aerosol model file",
    )

    template_file: Optional[FilePath] = Field(
        description="MODTRAN input template file",
    )
