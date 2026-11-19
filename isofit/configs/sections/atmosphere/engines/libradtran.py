"""
LibRadTran (Library for Radiative Transfer) engine configuration.

LibRadTran is a comprehensive open-source RT package supporting UV to
thermal infrared wavelengths. This module configures LibRadTran for
ISOFIT atmospheric correction.
"""

from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BeforeValidator, ConfigDict, DirectoryPath, Field, FilePath

from isofit.configs.base import SectionModel

from . import standardName


class LibRadTranConfig(SectionModel):
    """
    LibRadTran engine configuration.

    LibRadTran is an open-source RT package with multiple solvers and
    extensive atmospheric profile options. It supports UV through thermal
    infrared wavelengths.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import LibRadTranConfig
    >>> librad = LibRadTranConfig(
    ...     kb_alpha_1=1.3,
    ...     kb_alpha_2=0.2,
    ...     gg_set=0.7,
    ... )
    >>> librad.reptran_band_model, librad.kb_alpha_1
    ('coarse', 1.3)

    Notes
    -----
    LibRadTran is freely available from: http://www.libradtran.org/

    The REPTRAN band model provides efficient absorption calculations.
    Only the "coarse" REPTRAN band model is currently supported by
    ``isofit download libradtran``; the "medium" and "fine" models must be
    obtained separately from the LibRadTran website.

    LibRadTran offers multiple RT solvers including DISORT, SDISORT, and
    Monte Carlo options, configured in the template file.

    See Also
    --------
    ModtranConfig : Commercial RT engine alternative
    SixSConfig : Simpler open-source RT engine
    """

    section = "forward_model.atmosphere.engine"
    mixins = ("libradtran",)

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["libradtran"], BeforeValidator(standardName)] = Field(
        description='Engine identifier. Must be ``"libradtran"``',
    )

    base_dir: Optional[DirectoryPath] = Field(
        description="Base path to the LibRadTran installation directory",
        examples=["/opt/libradtran"],
    )

    band_model: Literal[
        "reptran coarse", "reptran medium", "reptran fine", "LOWTRAN", "KATO", "KATO2"
    ] = Field(
        default="reptran coarse",
        description=(
            "Molecular absorption band model and parameterization. Options: "
            "``reptran coarse`` (15 cm-1), ``reptran medium`` (5 cm-1), "
            "``reptran fine`` (1 cm-1), ``LOWTRAN`` (3-term fit from SBDART), "
            "``KATO``, and ``KATO2``. "
            "Please note it is not advised to use kato or kato2 in imaging spectroscopy retrievals, "
            "and these options are present only for efficient unit testing."
            "The default ``reptran coarse`` (or finer) is recommended for band_model."
        ),
        examples=["reptran coarse", "KATO2"],
    )

    kb_alpha_1: Optional[float] = Field(
        description=(
            "King-Byrne Angstrom parameter (alpha 1) for aerosol wavelength "
            "dependence. Setting to None uses the default aerosol profile"
        ),
        examples=[1.3],
    )

    kb_alpha_2: Optional[float] = Field(
        description=(
            "King-Byrne Angstrom parameter (alpha 2) for aerosol wavelength "
            "dependence. Setting to None uses the default aerosol profile"
        ),
        examples=[0.2],
    )

    gg_set: Optional[float] = Field(
        description=(
            "Constant asymmetry parameter (g) to represent the aerosol phase "
            "function. Setting to None uses the profile default"
        ),
        examples=[0.7],
    )

    gg_file: Optional[Path] = Field(
        description=(
            "Path to an asymmetry parameter file that overwrites the default "
            "profile with altitude-dependent values"
        ),
        examples=["gg_profile.txt"],
    )

    tau_file: Optional[Path] = Field(
        description=(
            "Path to an aerosol optical depth file that overwrites the default "
            "profile with altitude-dependent values"
        ),
        examples=["tau_profile.txt"],
    )

    ssa_file: Optional[Path] = Field(
        description=(
            "Path to a single scattering albedo file that overwrites the default "
            "aerosol absorption properties"
        ),
        examples=["ssa_profile.txt"],
    )

    ssa_scale: Optional[float] = Field(
        description="Scaling factor applied to the single scattering albedo",
        examples=[1.0],
    )

    moments_file: Optional[Path] = Field(
        description=(
            "Path to a phase function moments file that overwrites the default "
            "profile"
        ),
        examples=["moments.txt"],
    )

    template_file: Optional[FilePath] = Field(
        description="Template file used as the base configuration for libRadtran",
        examples=["libradtran_template.json"],
    )
