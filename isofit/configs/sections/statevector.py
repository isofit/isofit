"""
State vector configuration models for ISOFIT.

This module defines the structure of state vector elements used across
surface, atmosphere, and instrument models. State vectors represent the
parameters that ISOFIT optimizes during inversion.
"""

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from isofit.configs.base import SectionModel
from isofit.configs.utils.accessors import ElementAccessorMixin, StateVectorValueMixin


class StateVectorElementConfig(BaseModel):
    """
    Configuration for a single element in a state vector.

    A state vector element represents one parameter to be estimated during
    inversion. Each element has bounds, a scale factor, prior distribution
    parameters, and an initial guess value.

    Examples
    --------
    >>> from isofit.configs.sections.statevector import StateVectorElementConfig
    >>> element = StateVectorElementConfig(
    ...     bounds=[0.0, 1.0],
    ...     scale=0.5,
    ...     prior_mean=0.3,
    ...     prior_sigma=0.1,
    ...     init=0.25
    ... )
    >>> print(element.bounds)
    [0.0, 1.0]

    Notes
    -----
    Using np.nan as default allows ISOFIT to distinguish between "not set"
    and "explicitly set to 0". Parameters with nan values are handled
    specially during inversion setup.
    """

    bounds: list[float] = Field(
        default_factory=lambda: [np.nan, np.nan],
        description=(
            "Lower and upper bounds for the parameter during optimization. NaN "
            "bounds leave the parameter unbounded"
        ),
    )
    scale: float = Field(
        default=np.nan,
        description=(
            "Scale factor for normalizing the parameter during optimization, which "
            "improves numerical conditioning. NaN means no scaling"
        ),
    )
    prior_mean: float = Field(
        default=np.nan,
        description=(
            "Mean of the prior distribution for this parameter, used in Bayesian "
            "estimation. NaN means no prior"
        ),
    )
    prior_sigma: float = Field(
        default=np.nan,
        description=(
            "Standard deviation of the prior distribution for this parameter. NaN "
            "means no prior"
        ),
    )
    init: float = Field(
        default=np.nan,
        description=(
            "Initial guess for the parameter at the start of optimization. NaN means "
            "it is determined heuristically"
        ),
    )


class StateVectorConfig(SectionModel, ElementAccessorMixin, StateVectorValueMixin):
    """
    Base state vector configuration.

    This is an abstract base class for state vector configurations. Specific
    state vectors (SurfaceStateVectorConfig, InstrumentStateVectorConfig, etc.)
    inherit from this and add domain-specific state vector elements as fields.

    Each subclass declares its state vector elements as optional
    ``StateVectorElementConfig`` fields. The accessor methods below iterate over
    the elements that are actually set, so instrument/surface/atmosphere models
    can pull out bounds, scales, priors and initial values by name.

    Examples
    --------
    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "surface": {
    ...             "statevector": {"SURF_TEMP_K": {"bounds": [250.0, 350.0]}}
    ...         }
    ...     }
    ... })
    >>> sv = config.forward_model.surface.statevector
    >>> sv.get_all_bounds()
    [[250.0, 350.0]]

    See Also
    --------
    isofit.configs.sections.surface.SurfaceStateVectorConfig : Surface parameters
    isofit.configs.sections.instrument.InstrumentStateVectorConfig : Instrument parameters
    """
