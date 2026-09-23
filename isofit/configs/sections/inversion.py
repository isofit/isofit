"""
Inversion configuration for ISOFIT.

This module defines inversion settings including retrieval windows, optimization
parameters, MCMC configuration, and least-squares solver options.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class McmcConfig(BaseModel):
    """
    MCMC inversion configuration.

    Parameters for Markov Chain Monte Carlo inversion mode, which samples
    the posterior distribution to quantify retrieval uncertainty.

    Attributes
    ----------
    iterations : int
        Number of MCMC iterations to run. More iterations improve convergence
        but increase runtime. Default is 10000.
    burnin : int
        Number of burn-in iterations to discard before collecting samples.
        Should be long enough for chain to reach stationary distribution.
        Default is 200.
    regularizer : float
        Regularization parameter added to covariance matrix for numerical
        stability. Default is 1e-3.
    proposal_scaling : float
        Scaling factor for MCMC proposal distribution. Smaller values lead
        to higher acceptance rate but slower mixing. Default is 0.01.
    verbose : bool
        Enable verbose output during MCMC sampling. Default is True.
    restart_every : int
        Restart MCMC chain every N iterations to improve exploration.
        Default is 2000.

    Examples
    --------
    >>> from isofit.configs.sections.inversion import McmcConfig
    >>> mcmc = McmcConfig(iterations=20000, burnin=500, proposal_scaling=0.005)
    """

    iterations: int = Field(
        default=10000, description="Number of MCMC iterations to run"
    )

    burnin: int = Field(default=200, description="Number of burn-in iterations")

    regularizer: float = Field(default=1e-3, description="Regularization parameter")

    proposal_scaling: float = Field(
        default=0.01, description="Scaling factor for MCMC proposals"
    )

    verbose: bool = Field(default=True, description="Enable verbose output")

    restart_every: int = Field(default=2000, description="Restart interval for MCMC")


class LeastSquaresConfig(BaseModel):
    """
    Least squares inversion configuration.

    Parameters for scipy.optimize.least_squares solver used in standard
    inversion mode.

    Attributes
    ----------
    method : str
        Optimization method to use. Options: 'trf' (Trust Region Reflective),
        'dogbox', or 'lm' (Levenberg-Marquardt). Default is 'trf'.
    max_nfev : int
        Maximum number of forward model evaluations before termination.
        Lower values speed up processing but may terminate before convergence.
        Default is 20.
    xtol : float, optional
        Tolerance for termination by change of independent variables.
        If None, uses scipy default. Default is None.
    ftol : float
        Tolerance for termination by change of cost function. Smaller values
        require tighter convergence. Default is 0.01.
    gtol : float, optional
        Tolerance for termination by norm of gradient. If None, uses scipy
        default. Default is None.
    tr_solver : {"exact", "lsmr"}, optional
        Method for solving trust-region subproblems. "lsmr" is faster for
        large problems. Default is "lsmr".

    Examples
    --------
    >>> from isofit.configs.sections.inversion import LeastSquaresConfig
    >>> ls = LeastSquaresConfig(max_nfev=10, ftol=0.05, method="dogbox")

    See Also
    --------
    scipy.optimize.least_squares : Underlying solver documentation
    """

    method: str = Field(default="trf", description="Optimization method to use")

    max_nfev: int = Field(
        default=20,
        description="Maximum number of function evaluations before termination",
    )

    xtol: Optional[float] = Field(
        default=None,
        description="Tolerance for termination by change of independent variables",
    )

    ftol: float = Field(
        default=0.01, description="Tolerance for termination by change of cost function"
    )

    gtol: Optional[float] = Field(
        default=None, description="Tolerance for termination by norm of gradient"
    )

    tr_solver: Optional[Literal["exact", "lsmr"]] = Field(
        default="lsmr", description="Method for solving trust-region subproblems"
    )

    def get_config_options_as_dict(self):
        """
        Return the solver options as a plain ``dict`` keyed by field name.

        Tuples are converted to lists to match the historical behaviour of the
        old ``BaseConfigSection`` accessor. ``inversion/inverse.py`` feeds the
        result straight into ``scipy.optimize.least_squares`` as keyword
        arguments.
        """
        options = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, tuple):
                value = list(value)
            options[name] = value
        return options


class InversionConfig(BaseModel):
    """
    Inversion configuration for ISOFIT.

    Configures the retrieval algorithm including spectral windows, optimization
    method, MCMC parameters, and various inversion options.

    Attributes
    ----------
    windows : list[list[float]], optional
        Inversion retrieval windows to operate over, specified as list of
        [min_wavelength, max_wavelength] ranges in nanometers. Only wavelengths
        within these windows are used in the cost function. If None, uses all
        available wavelengths. Default is None.
    cressie_map_confidence : bool
        Use N. Cressie's alternate S_hat definition for more statistically-
        consistent posterior confidence intervals. Default is False.
    mcmc : McmcConfig
        MCMC parameters used when implementation.mode is "mcmc_inversion".
        Default uses McmcConfig defaults.
    integration_grid : dict
        Grid of inversion points for mode='grid' (advanced usage).
        Default is empty.
    priors_in_initial_guess : bool
        Use surface priors outside inversion windows during initial guess
        calculation. Improves initial guess quality. Default is True.
    inversion_grid_as_preseed : bool
        Treat inversion grid as seeds for optimization (True) or as fixed
        points to evaluate (False). Default is False.
    least_squares_params : LeastSquaresConfig
        Least squares parameters for core inversion solve in standard
        inversion mode. Default uses LeastSquaresConfig defaults.

    Examples
    --------
    >>> from isofit.configs.sections.inversion import InversionConfig, LeastSquaresConfig
    >>> inv = InversionConfig(
    ...     windows=[[400.0, 1300.0], [1450.0, 1780.0], [1950.0, 2450.0]],
    ...     least_squares_params=LeastSquaresConfig(max_nfev=15)
    ... )

    Notes
    -----
    Spectral windows are used to exclude atmospheric absorption features
    or noisy spectral regions from the retrieval. Common water vapor bands
    to avoid: 1350-1450 nm, 1780-1950 nm, >2500 nm.

    See Also
    --------
    McmcConfig : MCMC parameters
    LeastSquaresConfig : Least squares solver parameters
    """

    windows: Optional[List[List[float]]] = Field(
        default=None,
        description="Inversion retrieval windows to operate over (list of [min, max] wavelength ranges)",
    )

    cressie_map_confidence: bool = Field(
        default=False,
        description="Use N. Cressie's alternate S_hat definition for more statistically-consistent posterior confidence",
    )

    mcmc: McmcConfig = Field(
        default_factory=McmcConfig,
        description="MCMC parameters (only used if mode = mcmc)",
    )

    integration_grid: dict = Field(
        default_factory=dict,
        description="Grid of inversion points for mode='grid'",
    )

    priors_in_initial_guess: bool = Field(
        default=True,
        description="Use surface priors outside inversion windows during initial guess",
    )

    inversion_grid_as_preseed: bool = Field(
        default=False,
        description="Treat inversion grid as seeds for optimization (True) or fixed points (False)",
    )

    least_squares_params: LeastSquaresConfig = Field(
        default_factory=LeastSquaresConfig,
        description="Least squares parameters for core inversion solve",
    )

    @field_validator("windows")
    @classmethod
    def validate_windows(
        cls, v: Optional[List[List[float]]]
    ) -> Optional[List[List[float]]]:
        """
        Validate inversion windows format and ordering.

        Parameters
        ----------
        v : list[list[float]] or None
            Inversion windows to validate.

        Returns
        -------
        list[list[float]] or None
            The validated windows.

        Raises
        ------
        ValueError
            If windows are not a list of 2-element lists, or if wavelength
            ranges are not in ascending order.

        Notes
        -----
        Each window must be [min_wavelength, max_wavelength] with min < max.
        """
        if v is not None:
            for subset in v:
                if not isinstance(subset, list):
                    raise ValueError(
                        "windows parameter must be a list of lists of wavelength ranges"
                    )
                if len(subset) != 2:
                    raise ValueError(
                        f"Each window subset must have exactly 2 values, got {len(subset)}"
                    )
                if subset[0] > subset[1]:
                    raise ValueError(
                        f"In inversion window subset {subset}, wavelength ranges must be in order"
                    )
        return v
