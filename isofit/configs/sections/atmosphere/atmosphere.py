"""
Atmosphere configuration for ISOFIT.

This module defines atmospheric radiative transfer (RT) configuration including
RT engine selection, lookup table settings, and atmospheric state vector parameters.
"""

from pathlib import Path
from typing import Any, List, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from .engines import Engines
from .statevector import AtmosphereStateVectorConfig, AtmosphereUnknownsConfig


class AtmosphereConfig(BaseModel):
    """
    Atmosphere configuration.

    Configures atmospheric radiative transfer modeling including RT engine
    (MODTRAN, 6S, LibRadTran, sRTMnet, or prebuilt), lookup table generation
    and storage, and atmospheric state vector parameters.

    Attributes
    ----------
    engine : Engines
        Radiative transfer engine configuration. Defaults to prebuilt LUT
        (no RT code required).
    wavelength_file : Path, optional
        Optional path to wavelength file for high-resolution atmospheric
        calculations. If None, uses instrument wavelengths.
    lut_path : Path
        Path to the lookup table storage directory. Supports .zarr format
        for efficient multidimensional array storage.
    sim_path : Path
        Path to RT simulation output directory.
    lut_grid : dict[str, list[float]]
        Lookup table grid specification. Keys are atmospheric parameter names
        (e.g., "H2OSTR", "AOT550"), values are lists of sample points for
        that parameter. Empty dict means no LUT generation.
    lut_subset : dict[str, Any]
        Subset of lut_grid to use for processing. Allows using a smaller
        region of a large pre-computed LUT. Empty dict means use full grid.
        Accepts the legacy key ``lut_names`` as an alias for backward
        compatibility with older configs.
    rt_mode : {"rdn", "transm"}
        Atmospheric RT mode for LUT simulations. "transm" for transmittances
        (standard), "rdn" for reflected radiance. Default is "transm".
    statevector_names : list[str]
        Names of statevector elements to use with this atmospheric RT engine.
        Determines which atmospheric parameters are optimized during inversion.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere import AtmosphereConfig
    >>> from isofit.configs.sections.atmosphere.engines import ModtranConfig
    >>> atm = AtmosphereConfig(
    ...     engine=ModtranConfig(
    ...         aerosol_template_file="aerosol.txt",
    ...         template_file="template.json"
    ...     ),
    ...     lut_grid={
    ...         "H2OSTR": [0.5, 1.0, 1.5, 2.0, 2.5],
    ...         "AOT550": [0.05, 0.1, 0.2, 0.3]
    ...     },
    ...     statevector_names=["H2OSTR", "AOT550"]
    ... )

    Notes
    -----
    The lookup table approach enables fast atmospheric correction by
    precomputing RT simulations across a grid of atmospheric states.
    During inversion, ISOFIT interpolates this LUT rather than running
    the RT code for each pixel, reducing runtime by orders of magnitude.

    Commented-out validators in the code suggest that lut_grid validation
    (ensuring >= 2 unique values per dimension, sorting) may be re-enabled
    in future versions.

    See Also
    --------
    isofit.configs.sections.atmosphere.engines : Available RT engines
    """

    engine: Engines = Field(
        description="Radiative transfer engine; defaults to a prebuilt LUT",
    )

    wavelength_file: Path | None = Field(
        description="Optional path to wavelength file for high-res atmospheric calculations",
        examples=[""],
    )

    lut_path: Path = Field(
        description="Path to the look up table directory",
    )

    sim_path: Path = Field(
        description="Path to the look up table directory",
    )

    lut_grid: dict[str, list[float]] = Field(description="")

    lut_subset: dict[str, Any] = Field(
        validation_alias=AliasChoices("lut_subset", "lut_names"),
        description="Subset of the lut_grid to use (legacy alias: lut_names)",
    )

    rt_mode: Literal["rdn", "transm"] = Field(
        description="Atmospheric radiative transfer mode of LUT simulations: `transm` for transmittances, `rdn` for reflected radiance",
    )

    statevector_names: List[str] = Field(
        description="Names of the statevector elements to use with this atmospheric RT engine",
    )

    statevector: AtmosphereStateVectorConfig = Field(
        description="Atmospheric state vector elements",
    )

    unknowns: AtmosphereUnknownsConfig = Field(
        description="Radiative-transfer unknowns (unmodeled-variable uncertainties)",
    )

    configure_and_exit: bool = Field(
        description="Build/configure the RT engine and exit without running simulations",
    )

    interpolator_style: str | None = Field(
        description="LUT interpolation style; falls back to the instrument setting if unset",
    )

    wavelength_range: List[float] | None = Field(
        description="Optional [min, max] wavelength range to subset the LUT to after load",
    )

    multipart_transmittance: bool = Field(
        description="Apply triple-run diffuse & direct transmittance estimation",
    )

    irradiance_file: Path | None = Field(
        description="Solar irradiance file (used for RT and PACE-OCI SRF handling)",
    )
