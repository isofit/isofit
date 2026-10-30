"""
Forward model configuration for ISOFIT.

This module defines the forward model configuration, which combines surface,
atmosphere, and instrument models to simulate at-sensor radiance from surface
reflectance.
"""

from typing import Optional

from pydantic import Field, FilePath

from isofit.configs.base import SectionModel
from isofit.configs.sections.atmosphere import AtmosphereConfig
from isofit.configs.sections.instrument import InstrumentConfig
from isofit.configs.sections.surface import SurfaceConfig


class ForwardModelConfig(SectionModel):
    """
    Forward model configuration.

    The forward model combines surface, atmosphere, and instrument components
    to predict at-sensor radiance. During inversion, ISOFIT optimizes parameters
    in these components to match observed radiance.

    Examples
    --------
    >>> from isofit.configs.sections.forward_model import ForwardModelConfig
    >>> fm = ForwardModelConfig()
    >>> fm.atmosphere.engine.name, fm.instrument.SNR
    ('prebuilt', 500.0)

    Starting from the ``vswir`` base config, overriding nested sections with
    dicts:

    >>> fm = ForwardModelConfig(
    ...     base="vswir",
    ...     surface={"surface_file": "surface.mat"},
    ...     instrument={"SNR": 100.0},
    ... )
    >>> fm.atmosphere.engine.name, fm.surface.surface_category
    ('srtmnet', 'multicomponent_surface')
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

    section = "forward_model"

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
