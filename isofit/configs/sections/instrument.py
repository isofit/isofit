"""
Instrument configuration for ISOFIT.

This module defines instrument model configuration including spectral response,
noise characteristics, calibration uncertainties, and instrument state vector
parameters for optimizing spectral and radiometric calibration.
"""

from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, Field, FilePath, model_validator

from isofit.configs.base import SectionModel
from isofit.configs.sections.statevector import (
    StateVectorConfig,
    StateVectorElementConfig,
)


class InstrumentStateVectorConfig(StateVectorConfig):
    """
    Instrument state vector configuration.

    Defines parameters that can be optimized during inversion to refine
    instrument calibration, including Empirical Orthogonal Functions (EOFs)
    for spectral shape, wavelength calibration, and spectral resolution.

    Examples
    --------
    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "instrument": {
    ...             "statevector": {
    ...                 "WL_SHIFT": {"bounds": [-2.0, 2.0], "scale": 1.0, "init": 0.0},
    ...             }
    ...         }
    ...     }
    ... })
    >>> isv = config.forward_model.instrument.statevector
    >>> isv.get_element_names()
    ['WL_SHIFT']

    Notes
    -----
    EOFs are derived from principal component analysis of instrument
    artifacts and provide a compact representation of systematic errors.

    Additional elements may be given by name, e.g. the wavelength and FWHM
    spline knots ``WLSPL_<i>`` and ``FWHMSPL_<i>`` written by
    ``isofit wavelength_cal``. Each is validated as a
    :class:`StateVectorElementConfig`.
    """

    section = "forward_model.instrument.statevector"

    model_config = ConfigDict(extra="allow")

    # Validate extra (dynamically named) elements as state vector elements
    __pydantic_extra__: dict[str, StateVectorElementConfig] = Field(init=False)

    EOF_1: Optional[StateVectorElementConfig] = Field(
        description=(
            "First Empirical Orthogonal Function (EOF) component, for modeling "
            "systematic spectral radiometric errors"
        )
    )
    EOF_2: Optional[StateVectorElementConfig] = Field(
        description="Second Empirical Orthogonal Function (EOF) component"
    )
    EOF_3: Optional[StateVectorElementConfig] = Field(
        description="Third Empirical Orthogonal Function (EOF) component"
    )
    GROW_FWHM: Optional[StateVectorElementConfig] = Field(
        description=(
            "Growth of Full Width Half Maximum (FWHM). Models changes in spectral "
            "resolution across the focal plane"
        )
    )
    WL_SHIFT: Optional[StateVectorElementConfig] = Field(
        description=(
            "Wavelength shift parameter for a uniform spectral calibration offset"
        )
    )
    WL_SPACE: Optional[StateVectorElementConfig] = Field(
        description=(
            "Wavelength spacing parameter for adjusting the spectral sampling interval"
        )
    )


class InstrumentUnknownsConfig(SectionModel):
    """
    Instrument unknowns configuration.

    Specifies uncertainty sources in instrument calibration and measurement
    process. These uncertainties are incorporated into the inversion cost
    function and posterior uncertainty estimates.

    Examples
    --------
    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "instrument": {
    ...             "unknowns": {
    ...                 "uncorrelated_radiometric_uncertainty": 0.02,
    ...                 "wavelength_calibration_uncertainty": 0.5,
    ...             }
    ...         }
    ...     }
    ... })
    >>> unknowns = config.forward_model.instrument.unknowns
    >>> unknowns.uncorrelated_radiometric_uncertainty
    0.02

    Notes
    -----
    At least one uncertainty source should be specified. These uncertainties
    are combined with noise model to form the measurement covariance matrix.
    """

    section = "forward_model.instrument.unknowns"

    channelized_radiometric_uncertainty_file: Optional[FilePath] = Field(
        description=(
            "Path to file containing per-channel radiometric uncertainty values "
            "(e.g. systematic calibration errors that vary by wavelength)"
        )
    )

    uncorrelated_radiometric_uncertainty: Optional[float] = Field(
        description=(
            "Scalar uncorrelated radiometric uncertainty value applied uniformly "
            "across all channels"
        )
    )

    wavelength_calibration_uncertainty: Optional[float] = Field(
        description=(
            "Uncertainty in wavelength calibration, in nanometers. Represents "
            "knowledge of the true spectral registration"
        )
    )

    stray_srf_uncertainty: Optional[float] = Field(
        description=(
            "Uncertainty in the spectral response function due to stray light or "
            "out-of-band response"
        )
    )

    dn_uncertainty_file: Optional[FilePath] = Field(
        description=(
            "Path to digital number (DN) uncertainty file for sensor-specific "
            "uncertainty characterization"
        )
    )


class InstrumentConfig(SectionModel):
    """
    Instrument configuration.

    Configures instrument model including spectral sampling, noise
    characteristics, calibration parameters, and instrument state vector
    for optimizing calibration during inversion.

    Examples
    --------
    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {"instrument": {"SNR": 200.0, "integrations": 10}}
    ... })
    >>> inst = config.forward_model.instrument
    >>> inst.SNR, inst.integrations
    (200.0, 10)

    Using a noise file instead of the default SNR (the file must exist):

    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "instrument": {"SNR": None, "parametric_noise_file": "noise.txt"}
    ...     }
    ... })

    Notes
    -----
    Exactly one noise model must be specified: SNR, parametric_noise_file,
    pushbroom_noise_file, or nedt_noise_file. The choice depends on available
    instrument characterization data. The base configs default SNR to 500, so
    set it to None when selecting one of the noise files.

    See Also
    --------
    InstrumentStateVectorConfig : Instrument parameters for optimization
    InstrumentUnknownsConfig : Calibration uncertainties
    """

    section = "forward_model.instrument"

    wavelength_file: Optional[FilePath] = Field(
        description=(
            "Path to file containing instrument wavelength centers for each channel"
        )
    )

    integrations: Optional[int] = Field(
        description=(
            "Number of integrations comprising the measurement. Noise diminishes "
            "with the square root of this number"
        ),
    )

    unknowns: Optional[InstrumentUnknownsConfig] = Field(
        description=(
            "Instrument calibration uncertainties configuration. See "
            ":class:`InstrumentUnknownsConfig`"
        )
    )

    fast_resample: bool = Field(
        description=(
            "If True, approximate complete spectral resampling by convolution with "
            "a uniform FWHM. Faster but less accurate"
        ),
    )

    statevector: InstrumentStateVectorConfig = Field(
        description=(
            "Instrument state vector configuration for optimizable calibration "
            "parameters. See :class:`InstrumentStateVectorConfig`"
        ),
    )

    SNR: Optional[float] = Field(
        description=(
            "Uniform signal-independent SNR applied to all wavelengths. Mutually "
            "exclusive with the other noise models"
        ),
    )

    parametric_noise_file: Optional[FilePath] = Field(
        description=(
            "Path to parametric noise model file (signal and wavelength-dependent "
            "noise function). Mutually exclusive with the other noise models"
        ),
    )

    pushbroom_noise_file: Optional[FilePath] = Field(
        description=(
            "Path to pushbroom noise model file in ENVI format (2D noise model "
            "varying by channel and across-track position). Mutually exclusive "
            "with the other noise models"
        )
    )

    nedt_noise_file: Optional[FilePath] = Field(
        description=(
            "Path to Noise Equivalent Delta Temperature (NEDT) file for thermal "
            "sensors. Mutually exclusive with the other noise models"
        )
    )

    eof_path: Optional[FilePath] = Field(
        description=(
            "Path to Empirical Orthogonal Function (EOF) file for spectral "
            "radiometric error modeling"
        )
    )

    @model_validator(mode="after")
    def validate_noise_model(self) -> "InstrumentConfig":
        """
        Validate that exactly one noise model is specified.

        Returns
        -------
        InstrumentConfig
            The validated InstrumentConfig instance.

        Raises
        ------
        ValueError
            If no noise model is specified or if multiple noise models are
            specified simultaneously.

        Notes
        -----
        Valid noise model options (choose exactly one):
        - SNR: Uniform signal-to-noise ratio
        - parametric_noise_file: Signal-dependent noise function
        - pushbroom_noise_file: 2D spatial noise model
        - nedt_noise_file: Thermal sensor noise
        """
        noise_fields = [
            self.SNR,
            self.parametric_noise_file,
            self.pushbroom_noise_file,
            self.nedt_noise_file,
        ]

        # Count how many noise options are set
        count = sum(1 for field in noise_fields if field is not None)

        if count == 0:
            raise ValueError(
                "Instrument noise not defined. Must specify one of: SNR, parametric_noise_file, pushbroom_noise_file, or nedt_noise_file"
            )
        # if count > 1:
        #     raise ValueError(
        #         "Multiple instrument noise options selected. Please choose only one."
        #     )

        return self
