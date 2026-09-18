"""
Instrument configuration for ISOFIT.

This module defines instrument model configuration including spectral response,
noise characteristics, calibration uncertainties, and instrument state vector
parameters for optimizing spectral and radiometric calibration.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from isofit.configs.sections.statevector import StateVector, StateVectorElement
from isofit.configs.utils.validators import PathExists


class InstrumentStateVector(StateVector):
    """
    Instrument state vector configuration.

    Defines parameters that can be optimized during inversion to refine
    instrument calibration, including Empirical Orthogonal Functions (EOFs)
    for spectral shape, wavelength calibration, and spectral resolution.

    Attributes
    ----------
    EOF_1 : StateVectorElement, optional
        First Empirical Orthogonal Function for modeling systematic spectral
        radiometric errors.
    EOF_2 : StateVectorElement, optional
        Second EOF component.
    EOF_3 : StateVectorElement, optional
        Third EOF component.
    GROW_FWHM : StateVectorElement, optional
        Growth of Full Width Half Maximum. Models changes in spectral resolution
        across the focal plane.
    WL_SHIFT : StateVectorElement, optional
        Wavelength shift parameter for uniform spectral calibration offset.
    WL_SPACE : StateVectorElement, optional
        Wavelength spacing parameter for spectral sampling interval adjustment.

    Examples
    --------
    >>> from isofit.configs.sections.instrument import InstrumentStateVector
    >>> from isofit.configs.sections.statevector import StateVectorElement
    >>> isv = InstrumentStateVector()
    >>> isv.WL_SHIFT = StateVectorElement(
    ...     bounds=[-2.0, 2.0],
    ...     scale=1.0,
    ...     init=0.0
    ... )

    Notes
    -----
    EOFs are derived from principal component analysis of instrument
    artifacts and provide a compact representation of systematic errors.
    """

    EOF_1: Optional[StateVectorElement] = Field(
        default=None, description="First Empirical Orthogonal Function"
    )
    EOF_2: Optional[StateVectorElement] = Field(
        default=None, description="Second Empirical Orthogonal Function"
    )
    EOF_3: Optional[StateVectorElement] = Field(
        default=None, description="Third Empirical Orthogonal Function"
    )
    GROW_FWHM: Optional[StateVectorElement] = Field(
        default=None, description="Growth of Full Width Half Maximum"
    )
    WL_SHIFT: Optional[StateVectorElement] = Field(
        default=None, description="Wavelength shift parameter"
    )
    WL_SPACE: Optional[StateVectorElement] = Field(
        default=None, description="Wavelength spacing parameter"
    )


class InstrumentUnknowns(BaseModel):
    """
    Instrument unknowns configuration.

    Specifies uncertainty sources in instrument calibration and measurement
    process. These uncertainties are incorporated into the inversion cost
    function and posterior uncertainty estimates.

    Attributes
    ----------
    channelized_radiometric_uncertainty_file : PathExists, optional
        Path to file containing per-channel radiometric uncertainty values
        (e.g., systematic calibration errors that vary by wavelength).
    uncorrelated_radiometric_uncertainty : float, optional
        Scalar uncorrelated radiometric uncertainty value applied uniformly
        across all channels.
    wavelength_calibration_uncertainty : float, optional
        Uncertainty in wavelength calibration in nanometers. Represents
        knowledge of true spectral registration.
    stray_srf_uncertainty : float, optional
        Uncertainty in spectral response function due to stray light or
        out-of-band response.
    dn_uncertainty_file : PathExists, optional
        Path to digital number uncertainty file for sensor-specific
        uncertainty characterization.

    Examples
    --------
    >>> from isofit.configs.sections.instrument import InstrumentUnknowns
    >>> unknowns = InstrumentUnknowns(
    ...     uncorrelated_radiometric_uncertainty=0.02,
    ...     wavelength_calibration_uncertainty=0.5
    ... )

    Notes
    -----
    At least one uncertainty source should be specified. These uncertainties
    are combined with noise model to form the measurement covariance matrix.
    """

    channelized_radiometric_uncertainty_file: Optional[PathExists] = Field(
        default=None, description="Path to channelized radiometric uncertainty file"
    )

    uncorrelated_radiometric_uncertainty: Optional[float] = Field(
        default=None, description="Uncorrelated radiometric uncertainty value"
    )

    wavelength_calibration_uncertainty: Optional[float] = Field(
        default=None, description="Wavelength calibration uncertainty"
    )

    stray_srf_uncertainty: Optional[float] = Field(
        default=None, description="Stray spectral response function uncertainty"
    )

    dn_uncertainty_file: Optional[PathExists] = Field(
        default=None, description="Path to digital number uncertainty file"
    )


class Instrument(BaseModel):
    """
    Instrument configuration.

    Configures instrument model including spectral sampling, noise
    characteristics, calibration parameters, and instrument state vector
    for optimizing calibration during inversion.

    Attributes
    ----------
    wavelength_file : PathExists, optional
        Path to file containing instrument wavelength centers for each channel.
    integrations : int, optional
        Number of integrations comprising the measurement. Noise diminishes
        with square root of this number.
    unknowns : InstrumentUnknowns, optional
        Instrument calibration uncertainties configuration.
    fast_resample : bool
        If True, approximate complete spectral resampling by convolution with
        uniform FWHM. Faster but less accurate. Default is True.
    statevector : InstrumentStateVector
        Instrument state vector configuration for optimizable calibration
        parameters. Default is empty.
    SNR : float, optional
        Uniform signal-independent SNR applied to all wavelengths. Mutually
        exclusive with other noise models.
    parametric_noise_file : PathExists, optional
        Path to parametric noise model file (signal and wavelength-dependent
        noise function). Mutually exclusive with other noise models.
    pushbroom_noise_file : PathExists, optional
        Path to pushbroom noise model file in ENVI format (2D noise model
        varying by channel and across-track position). Mutually exclusive
        with other noise models.
    nedt_noise_file : PathExists, optional
        Path to Noise Equivalent Delta Temperature (NEDT) file for thermal
        sensors. Mutually exclusive with other noise models.
    eof_path : PathExists, optional
        Path to Empirical Orthogonal Function (EOF) file for spectral
        radiometric error modeling.

    Examples
    --------
    >>> from isofit.configs.sections.instrument import Instrument
    >>> inst = Instrument(
    ...     wavelength_file="wavelengths.txt",
    ...     SNR=200.0,
    ...     integrations=10
    ... )

    Notes
    -----
    Exactly one noise model must be specified: SNR, parametric_noise_file,
    pushbroom_noise_file, or nedt_noise_file. The choice depends on available
    instrument characterization data.

    See Also
    --------
    InstrumentStateVector : Instrument parameters for optimization
    InstrumentUnknowns : Calibration uncertainties
    """

    wavelength_file: Optional[PathExists] = Field(
        default=None, description="Path to wavelength file"
    )

    integrations: Optional[int] = Field(
        default=None,
        description="Number of integrations comprising the measurement. Noise diminishes with square root of this number.",
    )

    unknowns: Optional[InstrumentUnknowns] = Field(
        default=None, description="Instrument unknowns configuration"
    )

    fast_resample: bool = Field(
        default=True,
        description="Approximate complete resampling by convolution with uniform FWHM",
    )

    statevector: InstrumentStateVector = Field(
        default_factory=InstrumentStateVector,
        description="Instrument state vector configuration",
    )

    SNR: Optional[float] = Field(
        default=None,
        description="Uniform signal-independent SNR applied to all wavelengths",
    )

    parametric_noise_file: Optional[PathExists] = Field(
        default=None,
        description="Path to parametric noise file (signal and wavelength-dependent noise function)",
    )

    pushbroom_noise_file: Optional[PathExists] = Field(
        default=None, description="Path to pushbroom noise model file (ENVI format)"
    )

    nedt_noise_file: Optional[PathExists] = Field(
        default=None, description="Path to NEDT noise file"
    )

    eof_path: Optional[PathExists] = Field(
        default=None, description="Path to Empirical Orthogonal Function (EOF) file"
    )

    @model_validator(mode="after")
    def validate_noise_model(self) -> "Instrument":
        """
        Validate that exactly one noise model is specified.

        Returns
        -------
        Instrument
            The validated Instrument instance.

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
        if count > 1:
            raise ValueError(
                "Multiple instrument noise options selected. Please choose only one."
            )

        return self
