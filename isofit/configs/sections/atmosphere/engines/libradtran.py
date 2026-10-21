"""
LibRadTran (Library for Radiative Transfer) engine configuration.

LibRadTran is a comprehensive open-source RT package supporting UV to
thermal infrared wavelengths. This module configures LibRadTran for
ISOFIT atmospheric correction.
"""

from pathlib import Path
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


class LibRadTranConfig(BaseModel):
    """
    LibRadTran engine configuration.

    LibRadTran is an open-source RT package with multiple solvers and
    extensive atmospheric profile options. It supports UV through thermal
    infrared wavelengths.

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
        description='Engine identifier. Must be ``"libradtran"``'
    )

    base_dir: Optional[DirectoryPath] = Field(
        description="Base path to the LibRadTran installation directory",
    )

    reptran_band_model: Literal["coarse", "medium", "fine"] = Field(
        description=(
            "REPTRAN correlated-k band model resolution. Options: ``coarse`` "
            "(15 cm-1, fastest), ``medium`` (5 cm-1), ``fine`` (1 cm-1, most "
            "accurate but slower)"
        ),
        examples=["coarse", "medium", "fine"],
    )

    kb_alpha_1: Optional[float] = Field(
        description=(
            "King-Byrne Angstrom parameter (alpha 1) for aerosol wavelength "
            "dependence. Setting to None uses the default aerosol profile"
        )
    )

    kb_alpha_2: Optional[float] = Field(
        description=(
            "King-Byrne Angstrom parameter (alpha 2) for aerosol wavelength "
            "dependence. Setting to None uses the default aerosol profile"
        )
    )

    gg_set: Optional[float] = Field(
        description=(
            "Constant asymmetry parameter (g) to represent the aerosol phase "
            "function. Setting to None uses the profile default"
        )
    )

    gg_file: Optional[Path] = Field(
        description=(
            "Path to an asymmetry parameter file that overwrites the default "
            "profile with altitude-dependent values"
        )
    )

    tau_file: Optional[Path] = Field(
        description=(
            "Path to an aerosol optical depth file that overwrites the default "
            "profile with altitude-dependent values"
        )
    )

    ssa_file: Optional[Path] = Field(
        description=(
            "Path to a single scattering albedo file that overwrites the default "
            "aerosol absorption properties"
        ),
    )

    ssa_scale: Optional[float] = Field(
        description="Scaling factor applied to the single scattering albedo",
    )

    moments_file: Optional[Path] = Field(
        description=(
            "Path to a phase function moments file that overwrites the default "
            "profile"
        ),
    )

    template_file: Optional[FilePath] = Field(
        description="Template file used as the base configuration for libRadtran",
    )
