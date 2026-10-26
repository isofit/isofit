"""
MODTRAN radiative transfer engine configuration.

MODTRAN is a widely-used atmospheric RT code for visible through thermal infrared
wavelengths. This module configures MODTRAN 6.0 for ISOFIT atmospheric correction.
"""

from typing import Annotated, Literal, Optional

from pydantic import BeforeValidator, ConfigDict, DirectoryPath, Field, FilePath

from isofit.configs.base import SectionModel

from . import standardName


class ModtranConfig(SectionModel):
    """
    MODTRAN 6.0 engine configuration.

    MODTRAN (MODerate resolution atmospheric TRANsmission) is a physics-based
    RT code supporting VNIR to TIR wavelengths with comprehensive atmospheric
    absorption and scattering.

    Examples
    --------
    The installation directory and template files must exist:

    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "atmosphere": {
    ...             "engine": {
    ...                 "name": "modtran",
    ...                 "base_dir": "/opt/modtran6.0",
    ...                 "aerosol_template_file": "aerosol_template.json",
    ...                 "aerosol_model_file": "aerosol_model.txt",
    ...                 "template_file": "modtran_template.json",
    ...                 "multipart_transmittance": True,
    ...             }
    ...         }
    ...     }
    ... })
    >>> modtran = config.forward_model.atmosphere.engine

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

    section = "forward_model.atmosphere.engine"

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["modtran"], BeforeValidator(standardName)] = Field(
        description='Engine identifier. Must be "modtran" (case-insensitive)'
    )

    base_dir: Optional[DirectoryPath] = Field(
        description="Base path to the MODTRAN installation directory",
    )

    multipart_transmittance: bool = Field(
        description=(
            "Apply triple-run diffuse & direct transmittance estimation for "
            "improved accuracy. Increases runtime 3x"
        ),
    )

    use_tp7: bool = Field(
        description=(
            "Write TP7 outputs instead of the default CHN outputs. Only applies "
            "when ``multipart_transmittance`` is enabled"
        ),
    )

    aerosol_template_file: Optional[FilePath] = Field(
        description=(
            "Path to the aerosol template file defining aerosol optical "
            "properties and vertical profiles"
        ),
    )

    aerosol_model_file: Optional[FilePath] = Field(
        description=(
            "Path to the aerosol model file with wavelength-dependent properties"
        ),
    )

    template_file: Optional[FilePath] = Field(
        description=(
            "Path to the MODTRAN input template file (tape5 or JSON format) with "
            "the base atmospheric configuration"
        ),
    )
