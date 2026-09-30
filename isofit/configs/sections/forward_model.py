"""
Forward model configuration for ISOFIT.

This module defines the forward model configuration, which combines surface,
atmosphere, and instrument models to simulate at-sensor radiance from surface
reflectance. It also provides backward compatibility for legacy configuration
formats.
"""

import typing
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from isofit.configs.sections.atmosphere import AtmosphereConfig
from isofit.configs.sections.atmosphere.engines import Engines, standardName
from isofit.configs.sections.instrument import InstrumentConfig
from isofit.configs.sections.surface import SurfaceConfig
from isofit.configs.utils.validators import PathExists

# Legacy engine keys whose names changed in the new schema. Maps the old
# key (as it appears under a radiative_transfer_engines entry) to the field
# name on the engine config model.
_LEGACY_ENGINE_RENAMES = {
    "engine_name": "name",
    "engine_base_dir": "base_dir",
    "obs_file": "obs",
}

# Legacy engine keys that, in the new schema, live at the atmosphere level as
# well as (or instead of) the engine level. These are copied up to the
# atmosphere dict even when they also route to the engine.
_LEGACY_SHARED_KEYS = {"multipart_transmittance"}


def _engine_field_map() -> dict:
    """
    Map each standardized engine name to the set of its config field names.

    Introspects the ``Engines`` discriminated union so legacy-config routing
    stays in sync as engines are added or removed.

    Returns
    -------
    dict[str, set[str]]
        Mapping of standardized engine name to the set of field names on that
        engine's config model.
    """
    union, _discriminator = typing.get_args(Engines)
    fields = {}
    for member in typing.get_args(union):
        model, tag = typing.get_args(member)
        fields[tag.tag] = set(model.model_fields)
    return fields


class ForwardModelConfig(BaseModel):
    """
    Forward model configuration.

    The forward model combines surface, atmosphere, and instrument components
    to predict at-sensor radiance. During inversion, ISOFIT optimizes parameters
    in these components to match observed radiance.

    Attributes
    ----------
    surface : SurfaceConfig, optional
        Surface model configuration specifying reflectance model, priors,
        and surface parameters like temperature and glint.
    atmosphere : AtmosphereConfig, optional
        Atmospheric radiative transfer configuration including RT engine,
        lookup table settings, and atmospheric state vector parameters.
    instrument : InstrumentConfig, optional
        Instrument model configuration including spectral response, noise
        characteristics, and instrument state vector parameters.
    model_discrepancy_file : PathExists, optional
        Path to numpy-format covariance matrix representing systematic
        model-observation discrepancy. Used to account for forward model
        errors during inversion.

    Examples
    --------
    >>> from isofit.configs.sections import (
    ...     ForwardModelConfig,
    ...     SurfaceConfig,
    ...     InstrumentConfig,
    ... )
    >>> from isofit.configs.sections.atmosphere import AtmosphereConfig
    >>> fm = ForwardModelConfig(
    ...     surface=SurfaceConfig(surface_file="surface.mat"),
    ...     atmosphere=AtmosphereConfig(),
    ...     instrument=InstrumentConfig(wavelength_file="wavelengths.txt", SNR=100.0)
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
    isofit.configs.sections.surface.SurfaceConfig : Surface configuration
    isofit.configs.sections.atmosphere.AtmosphereConfig : Atmosphere configuration
    isofit.configs.sections.instrument.InstrumentConfig : Instrument configuration
    """

    surface: Optional[SurfaceConfig] = Field(
        default=None, description="Surface configuration"
    )

    atmosphere: Optional[AtmosphereConfig] = Field(
        default=None, description="Atmospheric radiative transfer configuration"
    )

    instrument: Optional[InstrumentConfig] = Field(
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

                # Apply legacy key renames (e.g. engine_name -> name) before
                # routing so keys land under their new field names.
                engine_in = {
                    _LEGACY_ENGINE_RENAMES.get(key, key): value
                    for key, value in first_engine.items()
                }

                # Route each legacy engine key to either the engine sub-config
                # or the atmosphere level. The engine is selected by its
                # (renamed) `name`; unknown/unnamed engines fall through to the
                # atmosphere default (prebuilt).
                name = engine_in.get("name")
                engine_fields = _engine_field_map()
                selected = (
                    engine_fields.get(standardName(name))
                    if isinstance(name, str)
                    else None
                )

                engine = {}
                atmosphere = {}
                for key, value in engine_in.items():
                    if selected is not None and key in selected:
                        engine[key] = value
                        if key in _LEGACY_SHARED_KEYS:
                            atmosphere.setdefault(key, value)
                    else:
                        # Not an engine field for the selected engine; treat as
                        # an atmosphere-level setting (lut_path, sim_path, etc.).
                        atmosphere.setdefault(key, value)

                if engine:
                    atmosphere["engine"] = engine

                # Hoist statevector / lut_grid / unknowns from the RT container
                for key in ("statevector", "lut_grid", "unknowns"):
                    if key in rt:
                        atmosphere.setdefault(key, rt[key])
                data = {**data, "atmosphere": atmosphere}
        return data
