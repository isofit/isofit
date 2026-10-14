"""
LibRadTran (Library for Radiative Transfer) engine configuration.

LibRadTran is a comprehensive open-source RT package supporting UV to
thermal infrared wavelengths. This module configures LibRadTran for
ISOFIT atmospheric correction.
"""

from pathlib import Path
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


class LibRadTranConfig(BaseModel):
    """
    LibRadTran engine configuration.

    LibRadTran is an open-source RT package with multiple solvers and
    extensive atmospheric profile options. It supports UV through thermal
    infrared wavelengths.

    Attributes
    ----------
    name : Literal["libradtran"]
        Engine identifier. Must be "libradtran".
    base_dir : DirectoryPath
        Base path to LibRadTran installation directory. The base configs
        default this to the ISOFIT ini ``libradtran``/``libradtran.version``
        path. Accepts the legacy key ``engine_base_dir``.
    reptran_band_model : {"coarse", "medium", "fine"}
        REPTRAN correlated-k band model resolution. "coarse" (15 cm⁻¹) is
        fastest, "fine" (1 cm⁻¹) is most accurate but slower. Default is "coarse".
    kb_alpha_1 : float
        King-Byrne Angstrom parameter (α₁) for aerosol wavelength dependence.
        Setting to None uses default aerosol profile.
    kb_alpha_2 : float
        King-Byrne Angstrom parameter (α₂) for aerosol wavelength dependence.
        Setting to None uses default aerosol profile.
    gg_set : float
        Constant asymmetry parameter (g) to represent aerosol phase function.
    gg_file : Path
        Path to asymmetry parameter file that overwrites default profile
        with altitude-dependent values.
    tau_file : Path
        Path to aerosol optical depth file that overwrites default profile
        with altitude-dependent values.
    ssa_file : Path
        Path to single scattering albedo file that overwrites default
        aerosol absorption properties.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import LibRadTranConfig
    >>> librad = LibRadTranConfig(
    ...     name="libradtran",
    ...     base_dir="/opt/libRadtran",
    ...     reptran_band_model="medium",
    ...     kb_alpha_1=1.3,
    ...     kb_alpha_2=0.2,
    ...     gg_set=0.7
    ... )

    Notes
    -----
    LibRadTran is freely available from: http://www.libradtran.org/

    The REPTRAN band model provides efficient correlated-k absorption
    calculations. For high spectral resolution applications, "fine" mode
    or line-by-line calculations may be needed.

    LibRadTran offers multiple RT solvers including DISORT, SDISORT, and
    Monte Carlo options, configured in the template file.

    See Also
    --------
    ModtranConfig : Commercial RT engine alternative
    SixSConfig : Simpler open-source RT engine
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["libradtran"], BeforeValidator(standardName)] = Field(
        validation_alias=NameAlias,
    )

    base_dir: Optional[DirectoryPath] = Field(
        validation_alias=AliasChoices("base_dir", "engine_base_dir"),
        description="Base path to engine directory (legacy alias: engine_base_dir)",
    )

    reptran_band_model: Literal["coarse", "medium", "fine"] = Field(
        description="REPTRAN band model. Options: coarse (15cm-1), medium (5 cm-1), fine (1 cm-1)",
        examples=["coarse", "medium", "fine"],
    )

    kb_alpha_1: Optional[float] = Field(
        description="King-Byrne Angstrom parameter (alpha 1). Setting to None uses default aerosol profile."
    )

    kb_alpha_2: Optional[float] = Field(
        description="King-Byrne Angstrom parameter (alpha 2). Setting to None uses default aerosol profile."
    )

    gg_set: Optional[float] = Field(
        description="Constant asymmetry parameter to represent aerosols. Setting to None uses the profile default."
    )

    gg_file: Optional[Path] = Field(
        description="Path to an asymmetry depth file that overwrites default profile"
    )

    tau_file: Optional[Path] = Field(
        description="Path to an optical depth file that overwrites default profile"
    )

    ssa_file: Optional[Path] = Field(
        description="Path to a single scattering albedo file that overwrites default profile",
    )

    ssa_scale: Optional[float] = Field(
        description="Scaling factor applied to the single scattering albedo",
    )

    moments_file: Optional[Path] = Field(
        description="Path to a phase function moments file that overwrites default profile",
    )

    template_file: Optional[FilePath] = Field(
        description="Template file used as the base configuration for libRadtran",
    )
