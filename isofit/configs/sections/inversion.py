"""
Inversion configuration for ISOFIT.

This module defines inversion settings including retrieval windows, optimization
parameters, MCMC configuration, and least-squares solver options.
"""

from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator

from isofit.configs.base import SectionModel


class McmcConfig(SectionModel):
    """
    MCMC inversion configuration.

    Parameters for Markov Chain Monte Carlo inversion mode, which samples
    the posterior distribution to quantify retrieval uncertainty.

    Examples
    --------
    >>> from isofit.configs.sections.inversion import McmcConfig
    >>> mcmc = McmcConfig(iterations=20000, proposal_scaling=0.005)
    >>> mcmc.iterations, mcmc.burnin
    (20000, 200)

    The ``mcmc`` base config uses a longer burn-in:

    >>> McmcConfig(base="mcmc").burnin
    2000
    """

    section = "implementation.inversion.mcmc"

    iterations: int = Field(
        description=(
            "Number of MCMC iterations to run. More iterations improve convergence "
            "but increase runtime"
        )
    )

    burnin: int = Field(
        description=(
            "Number of burn-in iterations to discard before collecting samples. "
            "Should be long enough for the chain to reach its stationary "
            "distribution"
        )
    )

    regularizer: float = Field(
        description=(
            "Regularization parameter added to the covariance matrix for numerical "
            "stability"
        )
    )

    proposal_scaling: float = Field(
        description=(
            "Scaling factor for the MCMC proposal distribution. Smaller values lead "
            "to a higher acceptance rate but slower mixing"
        )
    )

    verbose: bool = Field(description="Enable verbose output during MCMC sampling")

    restart_every: int = Field(
        description="Restart the MCMC chain every N iterations to improve exploration"
    )


class LeastSquaresConfig(SectionModel):
    """
    Least squares inversion configuration.

    Parameters for scipy.optimize.least_squares solver used in standard
    inversion mode.

    Examples
    --------
    >>> from isofit.configs.sections.inversion import LeastSquaresConfig
    >>> ls = LeastSquaresConfig()
    >>> ls.method, ls.max_nfev
    ('trf', 20)
    >>> ls = LeastSquaresConfig(method="dogbox", max_nfev=10, ftol=0.05)
    >>> ls.method, ls.max_nfev
    ('dogbox', 10)

    See Also
    --------
    scipy.optimize.least_squares : Underlying solver documentation
    """

    section = "implementation.inversion.least_squares_params"

    method: str = Field(
        description=(
            "Optimization method to use. Options: ``trf`` (Trust Region "
            "Reflective), ``dogbox``, or ``lm`` (Levenberg-Marquardt)"
        )
    )

    max_nfev: int = Field(
        description=(
            "Maximum number of forward model evaluations before termination. Lower "
            "values speed up processing but may terminate before convergence"
        ),
    )

    xtol: Optional[float] = Field(
        description=(
            "Tolerance for termination by change of independent variables. If "
            "None, uses the scipy default"
        ),
    )

    ftol: float = Field(
        description=(
            "Tolerance for termination by change of cost function. Smaller values "
            "require tighter convergence"
        )
    )

    gtol: Optional[float] = Field(
        description=(
            "Tolerance for termination by norm of gradient. If None, uses the scipy "
            "default"
        )
    )

    tr_solver: Optional[Literal["exact", "lsmr"]] = Field(
        description=(
            "Method for solving trust-region subproblems. ``lsmr`` is faster for "
            "large problems"
        )
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


class InversionConfig(SectionModel):
    """
    Inversion configuration for ISOFIT.

    Configures the retrieval algorithm including spectral windows, optimization
    method, MCMC parameters, and various inversion options.

    Examples
    --------
    >>> from isofit.configs.sections.inversion import InversionConfig
    >>> inv = InversionConfig(least_squares_params={"max_nfev": 15})
    >>> inv.windows
    [[350.0, 1360.0], [1410.0, 1800.0], [1970.0, 2500.0]]
    >>> inv.least_squares_params.max_nfev
    15

    The ``thermal`` base config adds a thermal-infrared window:

    >>> InversionConfig(base="thermal").windows
    [[380.0, 1325.0], [1435.0, 1770.0], [1965.0, 2500.0], [7500.0, 12000.0]]

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

    section = "implementation.inversion"

    windows: Optional[List[List[float]]] = Field(
        description=(
            "Inversion retrieval windows to operate over, specified as a list of "
            "``[min_wavelength, max_wavelength]`` ranges in nanometers (each with "
            "min < max). Only wavelengths within these windows are used in the "
            "cost function. If None, uses all available wavelengths"
        ),
    )

    cressie_map_confidence: bool = Field(
        description=(
            "Use N. Cressie's alternate S_hat definition for more "
            "statistically-consistent posterior confidence intervals"
        ),
    )

    mcmc: McmcConfig = Field(
        description=(
            "MCMC parameters, used only when ``implementation.mode`` is "
            "``mcmc_inversion``. See :class:`McmcConfig`"
        ),
    )

    integration_grid: dict = Field(
        description="Grid of inversion points for ``mode='grid'`` (advanced usage)",
    )

    priors_in_initial_guess: bool = Field(
        description=(
            "Use surface priors outside inversion windows during initial guess "
            "calculation. Improves initial guess quality"
        ),
    )

    inversion_grid_as_preseed: bool = Field(
        description=(
            "Treat the inversion grid as seeds for optimization (True) or as fixed "
            "points to evaluate (False)"
        ),
    )

    least_squares_params: LeastSquaresConfig = Field(
        description=(
            "Least squares parameters for the core inversion solve in standard "
            "inversion mode. See :class:`LeastSquaresConfig`"
        ),
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
