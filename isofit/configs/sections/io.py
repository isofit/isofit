"""
Input/output file configuration for ISOFIT.

This module defines Pydantic models for specifying input data files
(radiance, observation geometry, location, etc.) and output product files
(estimated reflectance, state vectors, uncertainties, etc.).
"""

from pathlib import Path
from typing import ClassVar, Optional

from pydantic import ConfigDict, Field, FilePath

from isofit.configs.base import SectionModel
from isofit.configs.utils.accessors import ElementAccessorMixin


class InputConfig(SectionModel, ElementAccessorMixin):
    """
    Input file configuration for ISOFIT.

    Specifies all input data files required for ISOFIT processing, including
    measured radiance, observation geometry, location data, and optional
    auxiliary inputs like background reflectance and radiometric corrections.

    Examples
    --------
    The base configs leave every input unset:

    >>> from isofit.configs.sections.io import InputConfig
    >>> InputConfig().measured_radiance_file is None
    True

    Supplying scene inputs on top of a base config (the files must exist):

    >>> inputs = InputConfig(
    ...     base="vswir",
    ...     measured_radiance_file="data/rdn.hdr",
    ...     obs_file="data/obs.hdr",
    ...     loc_file="data/loc.hdr",
    ... )

    Notes
    -----
    All file paths are validated to ensure they exist on the filesystem
    using the FilePath type. Paths can be absolute or relative to the
    configuration file directory (expansion happens in the loader).
    """

    section = "input"

    # Aliased fields (rdn/obs/loc) must also be settable by their full field
    # names, which is how the generated configs (e.g. apply_oe) write them.
    # Pydantic v2 restricts population to the alias unless this is enabled.
    model_config = ConfigDict(populate_by_name=True)

    measured_radiance_file: Optional[FilePath] = Field(
        alias="rdn",
        description=(
            "Input radiance file (.mat, .txt, or ENVI format) containing at-sensor "
            "radiance measurements for inverse modeling"
        ),
        examples=["data/rdn.hdr"],
    )

    reference_reflectance_file: Optional[FilePath] = Field(
        alias="rrf",
        description="Input reference reflectance file for radiometric calibration",
        examples=["data/reference_reflectance.hdr"],
    )

    reflectance_file: Optional[FilePath] = Field(
        alias="rfl",
        description=(
            "Input reflectance file for forward modeling (reflectance -> radiance). "
            "Required when ``implementation.mode`` is 'simulation'"
        ),
        examples=["data/rfl.hdr"],
    )

    obs_file: Optional[FilePath] = Field(
        alias="obs",
        description=(
            "Observation file with viewing and illumination geometry (solar zenith, "
            "view zenith, relative azimuth angles)"
        ),
        examples=["data/obs.hdr"],
    )

    glt_file: Optional[FilePath] = Field(
        alias="glt",
        description=(
            "Geographic Lookup Table (GLT) file with spatial offset information for "
            "orthorectified data"
        ),
        examples=["data/glt.hdr"],
    )

    loc_file: Optional[FilePath] = Field(
        alias="loc",
        description=(
            "Location file with latitude, longitude, and elevation for each pixel"
        ),
        examples=["data/loc.hdr"],
    )

    background_reflectance_file: Optional[Path] = Field(
        alias="brf",
        description=(
            "Background reflectance file for spatial inference and contextual "
            "atmospheric correction"
        ),
        examples=["data/background_reflectance.hdr"],
    )

    radiometry_correction_file: Optional[FilePath] = Field(
        alias="rcf",
        description="Radiometric correction file for systematic error correction",
        examples=["data/radiometry_correction.txt"],
    )

    skyview_factor_file: Optional[FilePath] = Field(
        alias="sff",
        description=(
            "Skyview factor file to modulate diffuse radiance contribution based on "
            "terrain occlusion"
        ),
        examples=["data/skyview_factor.hdr"],
    )


class OutputConfig(SectionModel):
    """
    Output file configuration for ISOFIT.

    Specifies output directory and all output product files that ISOFIT can
    generate, including estimated surface reflectance, state vectors, modeled
    radiance, atmospheric coefficients, uncertainties, and diagnostic products.

    Examples
    --------
    The ``default`` base config writes no products:

    >>> from isofit.configs.sections.io import OutputConfig
    >>> OutputConfig().get_output_files()[2]
    []

    The use-case base configs name their products relative to ``dir``:

    >>> output = OutputConfig(base="vswir", dir="results")
    >>> output.estimated_reflectance_file
    PosixPath('results/estimated_reflectance')
    >>> output.get_output_files()[2]
    ['estimated_reflectance_file', 'estimated_state_file', 'posterior_uncertainty_file']

    Notes
    -----
    Output file paths can be None if a particular product is not needed.
    Only specified outputs will be written, reducing disk usage and processing
    time for large datasets.
    """

    section = "output"

    dir: Path = Field(
        description="Output directory for ISOFIT", examples=["./isofit-output/"]
    )

    reset: bool = Field(
        description=(
            "If True, delete the output directory before starting if it exists. "
            "Leave False to preserve previous results"
        ),
        examples=[False],
    )

    estimated_state_file: Optional[Path] = Field(
        description=(
            "Output file for estimated state vector containing optimized parameter "
            "values for each pixel"
        ),
        examples=["isofit-output/estimated_state"],
    )

    estimated_reflectance_file: Optional[Path] = Field(
        description="Output file for estimated Lambertian surface reflectance",
        examples=["isofit-output/estimated_reflectance"],
    )

    estimated_emission_file: Optional[Path] = Field(
        description="Output file for estimated emitted radiance (thermal component)",
        examples=["isofit-output/estimated_emission"],
    )

    modeled_radiance_file: Optional[Path] = Field(
        description=(
            "Output file for forward-modeled at-sensor radiance using the estimated "
            "surface and atmosphere"
        ),
        examples=["isofit-output/modeled_radiance"],
    )

    apparent_reflectance_file: Optional[Path] = Field(
        description=(
            "Output file for apparent surface reflectance (radiance / downwelling "
            "irradiance) without atmospheric correction"
        ),
        examples=["isofit-output/apparent_reflectance"],
    )

    path_radiance_file: Optional[Path] = Field(
        description="Output file for path radiance contribution",
        examples=["isofit-output/path_radiance"],
    )

    simulated_measurement_file: Optional[Path] = Field(
        description="Output file for simulated at-sensor radiance in simulation mode",
        examples=["isofit-output/simulated_measurement"],
    )

    algebraic_inverse_file: Optional[Path] = Field(
        description="Output file for algebraic inverse solution",
        examples=["isofit-output/algebraic_inverse"],
    )

    atmospheric_coefficients_file: Optional[Path] = Field(
        description=(
            "Output file for atmospheric optical parameters (aerosol optical depth, "
            "water vapor, etc.)"
        ),
        examples=["isofit-output/atmospheric_coefficients"],
    )

    radiometry_correction_file: Optional[Path] = Field(
        description="Output file for radiometric correction factors",
        examples=["isofit-output/radiometry_correction"],
    )

    spectral_calibration_file: Optional[Path] = Field(
        description="Output file for spectral calibration (wavelength corrections)",
        examples=["isofit-output/spectral_calibration"],
    )

    posterior_uncertainty_file: Optional[Path] = Field(
        description=(
            "Output file for posterior uncertainty (diagonal of posterior covariance)"
        ),
        examples=["isofit-output/posterior_uncertainty"],
    )

    tilt_coefficients_file: Optional[Path] = Field(
        description=(
            "Output file for PCA coefficients of the ratio between DEM-tilted and "
            "flat reflectance. Requires ``forward_model.atmosphere.tilt_basis_file`` "
            "and a flat ``terrain_style``"
        ),
        examples=["isofit-output/tilt_coefficients"],
    )

    plot_surface_components: bool = Field(
        description="If True, generate diagnostic plots of surface components",
        examples=[False],
    )

    mcmc_samples_file: Optional[Path] = Field(
        description="Output file for MCMC samples when running in mcmc_inversion mode",
        examples=["isofit-output/mcmc_samples"],
    )

    # Header metadata (band-name key, ztitle, zrange) for each writable output
    # product, mirroring the old ``_<field>_header`` attributes. Only the fields
    # listed here are treated as output files by ``get_output_files``.
    _output_file_headers: ClassVar[dict[str, tuple[str, str, str]]] = {
        "estimated_state_file": (
            "statevector",
            "{State Parameter, Value}",
            "{}",
        ),
        "estimated_reflectance_file": (
            "wavelength",
            "{Wavelength (nm), Lambertian Reflectance}",
            "{0.0,1.0}",
        ),
        "estimated_emission_file": (
            "wavelength",
            "{Wavelength (nm), Emitted Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        ),
        "modeled_radiance_file": (
            "wavelength",
            "{Wavelength (nm), Modeled Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        ),
        "apparent_reflectance_file": (
            "wavelength",
            "{Wavelength (nm), Apparent Surface Reflectance}",
            "{}",
        ),
        "path_radiance_file": (
            "wavelength",
            "{Wavelength (nm), Path Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        ),
        "simulated_measurement_file": (
            "wavelength",
            "{Wavelength (nm), Simulated Radiance (uW nm-1 cm-2 sr-1)}",
            "{}",
        ),
        "algebraic_inverse_file": (
            "wavelength",
            "{Wavelength (nm), Apparent Surface Reflectance}",
            "{}",
        ),
        "atmospheric_coefficients_file": (
            "atm_coeffs",
            "{Wavelength (nm), Atmospheric Optical Parameters}",
            "{}",
        ),
        "radiometry_correction_file": (
            "wavelength",
            "{Wavelength (nm), Radiometric Correction Factors}",
            "{}",
        ),
        "spectral_calibration_file": ("wavelength", "{}", "{}"),
        "posterior_uncertainty_file": (
            "statevector",
            "{State Parameter, Value}",
            "{}",
        ),
        "tilt_coefficients_file": (
            "tilt_pcs",
            "{PCA Component, Tilt Ratio Coefficient}",
            "{}",
        ),
    }

    def get_all_output_file_names(self):
        """Return the names of all fields that represent writable outputs."""
        return list(self._output_file_headers)

    def get_output_files(self):
        """
        Return ``(elements, headers, names)`` for every output that is set.

        Only fields with an associated header (see ``_output_file_headers``)
        are considered, and those whose value is ``None`` are dropped. The
        surviving entries are sorted alphabetically by field name to match the
        historical ordering that ``fileio`` relies on.
        """
        pairs = []
        for name, header in self._output_file_headers.items():
            element = getattr(self, name)
            if element is not None:
                pairs.append((element, header, name))

        pairs.sort(key=lambda pair: pair[2])

        elements = [element for element, _, _ in pairs]
        headers = [header for _, header, _ in pairs]
        names = [name for _, _, name in pairs]
        return elements, headers, names
