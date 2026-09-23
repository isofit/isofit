"""
Input/output file configuration for ISOFIT.

This module defines Pydantic models for specifying input data files
(radiance, observation geometry, location, etc.) and output product files
(estimated reflectance, state vectors, uncertainties, etc.).
"""

from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

from isofit.configs.utils.accessors import ElementAccessorMixin
from isofit.configs.utils.validators import PathExists


class InputConfig(BaseModel, ElementAccessorMixin):
    """
    Input file configuration for ISOFIT.

    Specifies all input data files required for ISOFIT processing, including
    measured radiance, observation geometry, location data, and optional
    auxiliary inputs like background reflectance and radiometric corrections.

    Attributes
    ----------
    measured_radiance_file : PathExists, optional
        Input radiance file (.mat, .txt, or ENVI format) containing at-sensor
        radiance measurements for inverse modeling. Also accepts alias 'rdn'.
    reference_reflectance_file : PathExists, optional
        Input reference reflectance file for radiometric calibration.
    reflectance_file : PathExists, optional
        Input reflectance file for forward modeling (reflectance -> radiance).
        Required when implementation.mode is 'simulation'.
    obs_file : PathExists, optional
        Observation file with viewing and illumination geometry (solar zenith,
        view zenith, relative azimuth angles). Also accepts alias 'obs'.
    glt_file : PathExists, optional
        Geographic Lookup Table (GLT) file with spatial offset information
        for orthorectified data.
    loc_file : PathExists, optional
        Location file with latitude, longitude, and elevation for each pixel.
        Also accepts alias 'loc'.
    background_reflectance_file : PathExists, optional
        Background reflectance file for spatial inference and contextual
        atmospheric correction.
    radiometry_correction_file : PathExists, optional
        Radiometric correction file for systematic error correction.
    skyview_factor_file : PathExists, optional
        Skyview factor file to modulate diffuse radiance contribution based
        on terrain occlusion.

    Examples
    --------
    >>> from isofit.configs.sections.io import InputConfig
    >>> input_config = InputConfig(
    ...     measured_radiance_file="data/radiance.mat",
    ...     obs_file="data/obs.txt",
    ...     loc_file="data/location.txt"
    ... )

    Notes
    -----
    All file paths are validated to ensure they exist on the filesystem
    using the PathExists type. Paths can be absolute or relative to the
    configuration file directory (expansion happens in the loader).
    """

    measured_radiance_file: Optional[PathExists] = Field(
        default=None,
        alias="rdn",
        description="Input radiance file (.mat, .txt, or ENVI format) for inverse-modeling",
    )

    reference_reflectance_file: Optional[PathExists] = Field(
        default=None,
        description="Input reference reflectance file for radiometric calibration",
    )

    reflectance_file: Optional[PathExists] = Field(
        default=None,
        description="Input reflectance file for forward-modeling (reflectance -> radiance)",
    )

    obs_file: Optional[PathExists] = Field(
        default=None,
        alias="obs",
        description="Observation file with viewing/illumination geometry",
    )

    glt_file: Optional[PathExists] = Field(
        default=None, description="GLT file with spatial offset information"
    )

    loc_file: Optional[PathExists] = Field(
        default=None,
        alias="loc",
        description="Location file with lat, lon, and elevation",
    )

    background_reflectance_file: Optional[PathExists] = Field(
        default=None, description="Background reflectance file for spatial inference"
    )

    radiometry_correction_file: Optional[PathExists] = Field(
        default=None, description="Radiometric correction file for systematic errors"
    )

    skyview_factor_file: Optional[PathExists] = Field(
        default=None, description="Skyview factor file to modulate diffuse radiance"
    )


class OutputConfig(BaseModel):
    """
    Output file configuration for ISOFIT.

    Specifies output directory and all output product files that ISOFIT can
    generate, including estimated surface reflectance, state vectors, modeled
    radiance, atmospheric coefficients, uncertainties, and diagnostic products.

    Attributes
    ----------
    dir : Path
        Output directory for ISOFIT products. Default is './isofit-output/'.
    reset : bool
        If True, delete the output directory before starting if it exists.
        Default is False to preserve previous results.
    estimated_state_file : Path, optional
        Output file for estimated state vector containing optimized parameter
        values for each pixel.
    estimated_reflectance_file : Path, optional
        Output file for estimated Lambertian surface reflectance.
    estimated_emission_file : Path, optional
        Output file for estimated emitted radiance (thermal component).
    modeled_radiance_file : Path, optional
        Output file for forward-modeled at-sensor radiance using estimated
        surface and atmosphere.
    apparent_reflectance_file : Path, optional
        Output file for apparent surface reflectance (radiance / downwelling
        irradiance) without atmospheric correction.
    path_radiance_file : Path, optional
        Output file for path radiance contribution.
    simulated_measurement_file : Path, optional
        Output file for simulated at-sensor radiance in simulation mode.
    algebraic_inverse_file : Path, optional
        Output file for algebraic inverse solution.
    atmospheric_coefficients_file : Path, optional
        Output file for atmospheric optical parameters (aerosol optical depth,
        water vapor, etc.).
    radiometry_correction_file : Path, optional
        Output file for radiometric correction factors.
    spectral_calibration_file : Path, optional
        Output file for spectral calibration (wavelength corrections).
    posterior_uncertainty_file : Path, optional
        Output file for posterior uncertainty (diagonal of posterior covariance).
    plot_surface_components : bool
        If True, generate diagnostic plots of surface components. Default is False.
    mcmc_samples_file : Path, optional
        Output file for MCMC samples when running in mcmc_inversion mode.

    Examples
    --------
    >>> from isofit.configs.sections.io import OutputConfig
    >>> from pathlib import Path
    >>> output_config = OutputConfig(
    ...     dir=Path("results/"),
    ...     estimated_reflectance_file=Path("results/reflectance.dat"),
    ...     posterior_uncertainty_file=Path("results/uncertainty.dat")
    ... )

    Notes
    -----
    Output file paths can be None if a particular product is not needed.
    Only specified outputs will be written, reducing disk usage and processing
    time for large datasets.
    """

    dir: Path = Field(
        default=Path("./isofit-output/"), description="Output directory for ISOFIT"
    )

    reset: bool = Field(
        default=False,
        description="Delete output directory before starting if it exists",
    )

    estimated_state_file: Optional[Path] = Field(
        default=None, description="Output file for estimated state vector"
    )

    estimated_reflectance_file: Optional[Path] = Field(
        default=None, description="Output file for estimated Lambertian reflectance"
    )

    estimated_emission_file: Optional[Path] = Field(
        default=None, description="Output file for estimated emitted radiance"
    )

    modeled_radiance_file: Optional[Path] = Field(
        default=None, description="Output file for modeled radiance"
    )

    apparent_reflectance_file: Optional[Path] = Field(
        default=None, description="Output file for apparent surface reflectance"
    )

    path_radiance_file: Optional[Path] = Field(
        default=None, description="Output file for path radiance"
    )

    simulated_measurement_file: Optional[Path] = Field(
        default=None, description="Output file for simulated radiance"
    )

    algebraic_inverse_file: Optional[Path] = Field(
        default=None, description="Output file for algebraic inverse"
    )

    atmospheric_coefficients_file: Optional[Path] = Field(
        default=None, description="Output file for atmospheric optical parameters"
    )

    radiometry_correction_file: Optional[Path] = Field(
        default=None, description="Output file for radiometric correction factors"
    )

    spectral_calibration_file: Optional[Path] = Field(
        default=None, description="Output file for spectral calibration"
    )

    posterior_uncertainty_file: Optional[Path] = Field(
        default=None, description="Output file for posterior uncertainty"
    )

    plot_surface_components: bool = Field(
        default=False, description="Generate plots of surface components"
    )

    mcmc_samples_file: Optional[Path] = Field(
        default=None, description="Output file for MCMC samples"
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
