"""
LibRadTran (Library for Radiative Transfer) engine configuration.

LibRadTran is a comprehensive open-source RT package supporting UV to
thermal infrared wavelengths. This module configures LibRadTran for
ISOFIT atmospheric correction.
"""

from pathlib import Path
from typing import Annotated, Literal

from isofit.configs.utils.validators import PathExists
from isofit.data import env
from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

from . import standardName


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
    base_dir : PathExists
        Base path to LibRadTran installation directory. If not provided,
        falls back to environment configuration.
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

    name: Annotated[Literal["libradtran"], BeforeValidator(standardName)]

    base_dir: PathExists = Field(
        default=None,
        description="Base path to engine directory",
    )

    reptran_band_model: Literal["coarse", "medium", "fine"] = Field(
        default="coarse",
        description="REPTRAN band model. Options: coarse (15cm-1), medium (5 cm-1), fine (1 cm-1)",
        examples=["coarse", "medium", "fine"],
    )

    kb_alpha_1: float = Field(
        description="King-Byrne Angstrom parameter (alpha 1). Setting to None uses default aerosol profile."
    )

    kb_alpha_2: float = Field(
        description="King-Byrne Angstrom parameter (alpha 2). Setting to None uses default aerosol profile."
    )

    gg_set: float = Field(
        description="Constant asymmetry parameter to represent aerosols"
    )

    gg_file: Path = Field(
        description="Path to an asymmetry depth file that overwrites default profile"
    )

    tau_file: Path = Field(
        description="Path to an optical depth file that overwrites default profile"
    )

    ssa_file: Path = Field(
        description="Path to a single scattering albedo file that overwrites default profile"
    )

    ssa_file: Path = Field(
        description="Path to a phase function moments file that overwrites default profile"
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
        This allows LibRadTran installation path to be configured once in
        environment settings and reused across multiple configurations.
        """
        if base := data.get("base_dir"):
            env.changePath("libradtran", base)
        data["base_dir"] = env.libradtran

        return data
