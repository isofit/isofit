"""
Atmosphere configuration for ISOFIT.

This module defines atmospheric radiative transfer (RT) configuration including
RT engine selection, lookup table settings, and atmospheric state vector parameters.
"""

from pathlib import Path
from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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

    The lut_grid dimensions (and each dimension's grid points), lut_subset, and
    statevector_names are sorted on validation.

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
        description="Subset of the lut_grid to use",
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

    interpolator_style: Literal["rg", "mlg", "mlg_numba"] = Field(
        description="LUT interpolation style: `rg` (RegularGrid), `mlg` (multilinear grid), or `mlg_numba`",
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

    @field_validator("lut_grid")
    @classmethod
    def validate_lut_grid(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        """
        Validate that each LUT dimension can be interpolated over.

        Parameters
        ----------
        v : dict[str, list[float]]
            The LUT grid to validate.

        Returns
        -------
        dict[str, list[float]]
            The validated LUT grid.

        Raises
        ------
        ValueError
            If a dimension has fewer than 2 grid points or duplicate grid points.
        """
        for key, points in v.items():
            if len(points) < 2:
                raise ValueError(
                    f"lut_grid item {key} has less than the required 2 elements"
                )
            if len(set(points)) < len(points):
                raise ValueError(f"Detected duplicate values in lut_grid item {key}")
        return v

    @model_validator(mode="after")
    def sort_lut_dimensions(self) -> "AtmosphereConfig":
        """
        Sort the LUT dimensions by name, and each dimension's grid points.

        The LUT dimension order determines simulation filenames and LUT layout,
        so it must not depend on the key order of the config file.

        Returns
        -------
        AtmosphereConfig
            The validated config.
        """
        self.lut_grid = {
            key: sorted(self.lut_grid[key]) for key in sorted(self.lut_grid)
        }
        self.lut_subset = {key: self.lut_subset[key] for key in sorted(self.lut_subset)}
        self.statevector_names = sorted(self.statevector_names)
        return self
