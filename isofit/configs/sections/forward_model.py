"""
Forward model configuration for ISOFIT.

This module defines the forward model configuration, which combines surface,
atmosphere, and instrument models to simulate at-sensor radiance from surface
reflectance. It also provides backward compatibility for legacy configuration
formats.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from isofit.configs.sections.atmosphere import Atmosphere
from isofit.configs.sections.instrument import Instrument
from isofit.configs.sections.surface import Surface
from isofit.configs.utils.validators import PathExists


class ForwardModel(BaseModel):
    """
    Forward model configuration.

    The forward model combines surface, atmosphere, and instrument components
    to predict at-sensor radiance. During inversion, ISOFIT optimizes parameters
    in these components to match observed radiance.

    Attributes
    ----------
    surface : Surface, optional
        Surface model configuration specifying reflectance model, priors,
        and surface parameters like temperature and glint.
    atmosphere : Atmosphere, optional
        Atmospheric radiative transfer configuration including RT engine,
        lookup table settings, and atmospheric state vector parameters.
    instrument : Instrument, optional
        Instrument model configuration including spectral response, noise
        characteristics, and instrument state vector parameters.
    model_discrepancy_file : PathExists, optional
        Path to numpy-format covariance matrix representing systematic
        model-observation discrepancy. Used to account for forward model
        errors during inversion.

    Examples
    --------
    >>> from isofit.configs.sections import ForwardModel, Surface, Instrument
    >>> from isofit.configs.sections.atmosphere import Atmosphere
    >>> fm = ForwardModel(
    ...     surface=Surface(surface_file="surface.mat"),
    ...     atmosphere=Atmosphere(),
    ...     instrument=Instrument(wavelength_file="wavelengths.txt", SNR=100.0)
    ... )

    Notes
    -----
    The forward model is the core physics engine of ISOFIT. It must be
    correctly configured to match the sensor and scene characteristics
    for accurate retrievals.

    This class includes a validator for backward compatibility with legacy
    configuration formats that used nested radiative_transfer structures.

    See Also
    --------
    isofit.configs.sections.surface.Surface : Surface configuration
    isofit.configs.sections.atmosphere.Atmosphere : Atmosphere configuration
    isofit.configs.sections.instrument.Instrument : Instrument configuration
    """

    surface: Optional[Surface] = Field(
        default=None, description="Surface configuration"
    )

    atmosphere: Optional[Atmosphere] = Field(
        default=None, description="Atmospheric radiative transfer configuration"
    )

    instrument: Optional[Instrument] = Field(
        default=None, description="Instrument configuration"
    )

    model_discrepancy_file: Optional[PathExists] = Field(
        default=None,
        description="Path to numpy-format covariance matrix for model discrepancy",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_format(cls, data: dict) -> dict:
        """
        Handle backward compatibility for old config format.

        Legacy ISOFIT configurations used a nested structure with
        forward_model.radiative_transfer containing radiative_transfer_engines.
        This validator flattens that structure to the new forward_model.atmosphere
        format.

        Parameters
        ----------
        data : dict
            Raw configuration data before validation.

        Returns
        -------
        dict
            Transformed configuration data with legacy structure converted
            to new format.

        Notes
        -----
        Legacy format:
            forward_model:
                radiative_transfer:
                    radiative_transfer_engines:
                        vswir:
                            engine: "modtran"
                            ...
                    statevector: {...}

        New format:
            forward_model:
                atmosphere:
                    engine: "modtran"
                    statevector: {...}
        """
        # Old configs used forward_model.radiative_transfer with nested radiative_transfer_engines
        # Flatten the first engine into forward_model.atmosphere
        if isinstance(data, dict):
            if "radiative_transfer" in data and "atmosphere" not in data:
                rt = data["radiative_transfer"]
                engines = rt.get("radiative_transfer_engines", {})
                # Take the first engine (usually the only one, e.g. "vswir")
                first_engine = next(iter(engines.values()), {}) if engines else {}
                atmosphere = {**first_engine}
                # Hoist statevector / lut_grid / unknowns from the RT container
                for key in ("statevector", "lut_grid", "unknowns"):
                    if key in rt:
                        atmosphere.setdefault(key, rt[key])
                data = {**data, "atmosphere": atmosphere}
        return data
