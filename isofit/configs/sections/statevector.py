"""
State vector configuration models for ISOFIT.

This module defines the structure of state vector elements used across
surface, atmosphere, and instrument models. State vectors represent the
parameters that ISOFIT optimizes during inversion.
"""

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from isofit.configs.utils.accessors import ElementAccessorMixin


class StateVectorElementConfig(BaseModel):
    """
    Configuration for a single element in a state vector.

    A state vector element represents one parameter to be estimated during
    inversion. Each element has bounds, a scale factor, prior distribution
    parameters, and an initial guess value.

    Attributes
    ----------
    bounds : list[float]
        Lower and upper bounds for the parameter during optimization.
        Default is [nan, nan] for unbounded parameters.
    scale : float
        Scale factor for normalizing the parameter during optimization.
        Improves numerical conditioning. Default is nan (no scaling).
    prior_mean : float
        Mean of the prior distribution for this parameter. Used in
        Bayesian estimation. Default is nan (no prior).
    prior_sigma : float
        Standard deviation of the prior distribution. Default is nan (no prior).
    init : float
        Initial guess value for the parameter at the start of optimization.
        Default is nan (will be determined heuristically).

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
        description="Bounds for the state vector element",
    )
    scale: float = Field(
        default=np.nan, description="Scale for the state vector element"
    )
    prior_mean: float = Field(
        default=np.nan, description="Prior mean for the state vector element"
    )
    prior_sigma: float = Field(
        default=np.nan, description="Prior sigma for the state vector element"
    )
    init: float = Field(
        default=np.nan, description="Initial value for the state vector element"
    )


class StateVectorConfig(BaseModel, ElementAccessorMixin):
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
    >>> from isofit.configs.sections.surface import SurfaceStateVectorConfig
    >>> from isofit.configs.sections.statevector import StateVectorElementConfig
    >>> sv = SurfaceStateVectorConfig()
    >>> sv.SURF_TEMP_K = StateVectorElementConfig(bounds=[250.0, 350.0])
    >>> sv.get_all_bounds()
    [[250.0, 350.0]]

    See Also
    --------
    isofit.configs.sections.surface.SurfaceStateVectorConfig : Surface parameters
    isofit.configs.sections.instrument.InstrumentStateVectorConfig : Instrument parameters
    """

    def get_all_bounds(self):
        """Return the ``bounds`` of every set element, ordered by name."""
        return [element.bounds for element, _ in zip(*self.get_elements())]

    def get_all_scales(self):
        """Return the ``scale`` of every set element, ordered by name."""
        return [element.scale for element, _ in zip(*self.get_elements())]

    def get_all_inits(self):
        """Return the ``init`` value of every set element, ordered by name."""
        return [element.init for element, _ in zip(*self.get_elements())]

    def get_all_prior_means(self):
        """Return the ``prior_mean`` of every set element, ordered by name."""
        return [element.prior_mean for element, _ in zip(*self.get_elements())]

    def get_all_prior_sigmas(self):
        """Return the ``prior_sigma`` of every set element, ordered by name."""
        return [element.prior_sigma for element, _ in zip(*self.get_elements())]
