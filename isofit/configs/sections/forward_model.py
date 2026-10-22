"""
Forward model configuration for ISOFIT.

This module defines the forward model configuration, which combines surface,
atmosphere, and instrument models to simulate at-sensor radiance from surface
reflectance.
"""

from typing import Optional

from pydantic import BaseModel, Field, FilePath

from isofit.configs.sections.atmosphere import AtmosphereConfig
from isofit.configs.sections.instrument import InstrumentConfig
from isofit.configs.sections.surface import SurfaceConfig


class ForwardModelConfig(BaseModel):
    """
    Forward model configuration.

    The forward model combines surface, atmosphere, and instrument components
    to predict at-sensor radiance. During inversion, ISOFIT optimizes parameters
    in these components to match observed radiance.

    Examples
    --------
    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "surface": {
    ...             "surface_file": "surface.mat",
    ...             "surface_category": "multicomponent_surface",
    ...         },
    ...         "atmosphere": {"engine": {"name": "prebuilt"}},
    ...         "instrument": {"SNR": 100.0},
    ...     }
    ... })
    >>> fm = config.forward_model
    >>> fm.instrument.SNR
    100.0

    Notes
    -----
    The forward model is the core physics engine of ISOFIT. It must be
    correctly configured to match the sensor and scene characteristics
    for accurate retrievals.

    Configs using the legacy ``radiative_transfer`` layout are converted by
    :mod:`isofit.configs.legacy` before validation.

    See Also
    --------
    isofit.configs.sections.surface.SurfaceConfig : Surface configuration
    isofit.configs.sections.atmosphere.AtmosphereConfig : Atmosphere configuration
    isofit.configs.sections.instrument.InstrumentConfig : Instrument configuration
    """

    surface: Optional[SurfaceConfig] = Field(
        description=(
            "Surface model configuration specifying the reflectance model, priors, "
            "and surface parameters like temperature and glint"
        )
    )

    atmosphere: Optional[AtmosphereConfig] = Field(
        description=(
            "Atmospheric radiative transfer configuration including the RT engine, "
            "lookup table settings, and atmospheric state vector parameters"
        )
    )

    instrument: Optional[InstrumentConfig] = Field(
        description=(
            "Instrument model configuration including spectral response, noise "
            "characteristics, and instrument state vector parameters"
        )
    )

    model_discrepancy_file: Optional[FilePath] = Field(
        description=(
            "Path to a numpy-format covariance matrix representing systematic "
            "model-observation discrepancy. Used to account for forward model "
            "errors during inversion"
        ),
    )
