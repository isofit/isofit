"""
State vector configuration models for ISOFIT.

This module defines the structure of state vector elements used across
surface, atmosphere, and instrument models. State vectors represent the
parameters that ISOFIT optimizes during inversion.
"""

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class StateVectorElement(BaseModel):
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
    >>> from isofit.configs.sections.statevector import StateVectorElement
    >>> element = StateVectorElement(
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


class StateVector(BaseModel):
    """
    Base state vector configuration.

    This is an abstract base class for state vector configurations. Specific
    state vectors (SurfaceStateVector, InstrumentStateVector, etc.) inherit
    from this and add domain-specific state vector elements as fields.

    Examples
    --------
    >>> from isofit.configs.sections.surface import SurfaceStateVector
    >>> from isofit.configs.sections.statevector import StateVectorElement
    >>> sv = SurfaceStateVector()
    >>> sv.SURF_TEMP_K = StateVectorElement(bounds=[250.0, 350.0])

    See Also
    --------
    isofit.configs.sections.surface.SurfaceStateVector : Surface parameters
    isofit.configs.sections.instrument.InstrumentStateVector : Instrument parameters
    """

    pass
