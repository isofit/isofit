"""
Atmosphere configuration for ISOFIT.

This module defines atmospheric radiative transfer (RT) configuration including
RT engine selection, lookup table settings, and atmospheric state vector parameters.
"""

from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .engines import Engines, Prebuilt


class Atmosphere(BaseModel):
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
        for efficient multidimensional array storage. Default uses interpolated
        string "${output.dir}/lut/lut.zarr".
    sim_path : Path
        Path to RT simulation output directory. Default uses interpolated
        string "${output.dir}/lut/sims".
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
    lut_names : list[str], optional
        Names of dimensions in the LUT. Provided for backward compatibility
        with older configs. Prefer using lut_grid keys.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere import Atmosphere
    >>> from isofit.configs.sections.atmosphere.engines import Modtran
    >>> atm = Atmosphere(
    ...     engine=Modtran(
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
        default_factory=lambda: Prebuilt(name="prebuilt"),
        description="Radiative transfer engine; defaults to a prebuilt LUT",
    )

    wavelength_file: Path | None = Field(
        default=None,
        description="Optional path to wavelength file for high-res atmospheric calculations",
        examples=[""],
    )

    lut_path: Path = Field(
        default="${output.dir}/lut/lut.zarr",
        description="Path to the look up table directory",
    )

    sim_path: Path = Field(
        default="${output.dir}/lut/sims",
        description="Path to the look up table directory",
    )

    lut_grid: dict[str, list[float]] = Field(default_factory=dict, description="")

    lut_subset: dict[str, Any] = Field(
        default_factory=dict, description="Subset of the lut_grid to use"
    )

    rt_mode: Literal["rdn", "transm"] = Field(
        default="transm",
        description="Atmospheric radiative transfer mode of LUT simulations: `transm` for transmittances, `rdn` for reflected radiance",
    )

    statevector_names: List[str] = Field(
        default_factory=list,
        description="Names of the statevector elements to use with this atmospheric RT engine",
    )

    lut_names: Optional[List[str]] = Field(
        default=None,
        description="Names of dimensions in the LUT (for backward compatibility)",
    )

    # @field_validator("lut_grid")
    # @classmethod
    # def _validate_lut_grid(
    #     cls, grid: dict[str, list[float]] | None
    # ) -> dict[str, list[float]] | None:
    #     """
    #     Each grid point must have at least 2 unique values. Values are sorted
    #     ascending and the grid is sorted alphabetically by key.
    #     """
    #     if grid is None:
    #         return grid
    #
    #     for key, values in grid.items():
    #         if len(values) < 2:
    #             raise ValueError(
    #                 f"lut_grid item {key!r} has fewer than the required 2 elements"
    #             )
    #         if len(set(values)) < len(values):
    #             raise ValueError(f"Detected duplicate values in lut_grid item {key!r}")
    #
    #     return {key: sorted(grid[key]) for key in sorted(grid)}
    #
    # @model_validator(mode="after")
    # def _validate_lut_subset(self) -> "Atmosphere":
    #     """
    #     lut_subset (formerly lut_names) must be a subset of the lut_grid keys.
    #     Sorted alphabetically by key to match lut_grid ordering.
    #     """
    #     if self.lut_subset is None:
    #         return self
    #
    #     if self.lut_grid is None:
    #         raise ValueError("lut_subset was provided but lut_grid is not set")
    #
    #     extra = set(self.lut_subset) - set(self.lut_grid)
    #     if extra:
    #         raise ValueError(
    #             f"lut_subset keys must be a subset of lut_grid keys; "
    #             f"unknown keys: {sorted(extra)}"
    #         )
    #
    #     self.lut_subset = {key: self.lut_subset[key] for key in sorted(self.lut_subset)}
    #     return self
