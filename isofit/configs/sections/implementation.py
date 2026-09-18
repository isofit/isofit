"""
Implementation configuration for ISOFIT.

This module defines runtime implementation settings including operating mode,
parallelization parameters, I/O buffering, and Ray distributed computing
configuration.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from isofit import __version__
from isofit.configs.sections.inversion import InversionConfig


class ImplementationConfig(BaseModel):
    """
    Implementation configuration for ISOFIT runtime.

    Configures how ISOFIT executes, including operating mode (inversion,
    MCMC, or simulation), parallelization strategy using Ray, I/O buffering,
    and various runtime optimizations.

    Attributes
    ----------
    mode : {"inversion", "mcmc_inversion", "simulation"}
        Operating mode for ISOFIT. "inversion" performs standard retrieval,
        "mcmc_inversion" uses Markov Chain Monte Carlo sampling for uncertainty
        quantification, "simulation" performs forward modeling only.
        Default is "inversion".
    inversion : InversionConfig, optional
        Inversion configuration. Required for inversion and mcmc_inversion modes.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing. If None, uses all
        available cores. Default is None.
    task_inflation_factor : int
        Submit task_inflation_factor * n_cores tasks to Ray to improve load
        balancing. Higher values improve CPU utilization but increase overhead.
        Default is 10.
    ip_head : str, optional
        Ray parameter: IP address and port of Ray head node for multi-node
        distributed runs (e.g., "192.168.1.1:6379"). Must be specified with
        redis_password. Default is None (single node).
    redis_password : str, optional
        Ray parameter: Redis password for multi-node runs. Must be specified
        with ip_head. Default is None.
    ray_include_dashboard : bool
        Ray parameter: Enable Ray dashboard for monitoring. Default is False.
    ray_temp_dir : str
        Ray temporary directory for object store and logs. Useful on multiuser
        systems to isolate Ray instances. Default is "/tmp/ray".
    ray_ignore_reinit_error : bool
        Tell Ray to ignore re-initialization errors. Convenient for running
        multiple ISOFIT instances sequentially. Default is True.
    io_buffer_size : int
        Size of chunks to read/process/write in number of spectra. Larger values
        improve I/O efficiency but increase memory usage. Default is 100.
    max_hash_table_size : int
        Maximum size of inversion hash tables for caching forward model results.
        Default is 50.
    per_pixel_heuristic_prior : bool
        Use per-pixel heuristic prior (True) or image-wide universal value (False).
        Per-pixel priors adapt to local scene characteristics. Default is False.
    debug_mode : bool
        Run in debug mode, which circumvents Ray parallelization for easier
        debugging. Default is False.
    isofit_version : str
        ISOFIT version string. Automatically set to current version.

    Examples
    --------
    >>> from isofit.configs.sections import ImplementationConfig, InversionConfig
    >>> impl = ImplementationConfig(
    ...     mode="inversion",
    ...     n_cores=8,
    ...     inversion=InversionConfig(),
    ...     io_buffer_size=200
    ... )

    Notes
    -----
    For multi-node distributed processing, both ip_head and redis_password
    must be set. The head node should be started separately with:

    .. code-block:: bash

        ray start --head --port=6379 --redis-password=<password>

    Worker nodes connect with:

    .. code-block:: bash

        ray start --address=<ip_head> --redis-password=<password>

    See Also
    --------
    isofit.configs.sections.inversion.InversionConfig : Inversion configuration
    """

    mode: Literal["inversion", "mcmc_inversion", "simulation"] = Field(
        default="inversion", description="Operating mode for ISOFIT"
    )

    inversion: Optional[InversionConfig] = Field(
        default=None, description="Inversion configuration"
    )

    n_cores: Optional[int] = Field(default=None, description="Number of cores to use")

    task_inflation_factor: int = Field(
        default=10, description="Submit task_inflation_factor*n_cores number of tasks"
    )

    ip_head: Optional[str] = Field(
        default=None, description="Ray parameter: IP-head for multi-node runs"
    )

    redis_password: Optional[str] = Field(
        default=None, description="Ray parameter: Redis-password for multi-node runs"
    )

    ray_include_dashboard: bool = Field(
        default=False, description="Ray parameter: Include dashboard"
    )

    ray_temp_dir: str = Field(
        default="/tmp/ray",
        description="Ray temporary directory (useful for multiuser systems)",
    )

    ray_ignore_reinit_error: bool = Field(
        default=True,
        description="Tell ray to ignore re-initialization (convenient for multiple ISOFIT instances)",
    )

    io_buffer_size: int = Field(
        default=100,
        description="Size of chunks to read/process/write (in number of spectra)",
    )

    max_hash_table_size: int = Field(
        default=50, description="Maximum size of inversion hash tables"
    )

    per_pixel_heuristic_prior: bool = Field(
        default=False,
        description="Use per-pixel heuristic prior (True) or image-wide universal value (False)",
    )

    debug_mode: bool = Field(
        default=False, description="Run in debug mode (circumvents ray)"
    )

    isofit_version: str = Field(default=__version__, description="ISOFIT version used")

    @model_validator(mode="after")
    def validate_implementation(self) -> "ImplementationConfig":
        """
        Validate implementation configuration.

        Ensures that required fields are set for the selected mode and that
        multi-node Ray parameters are specified together.

        Returns
        -------
        ImplementationConfig
            The validated ImplementationConfig instance.

        Raises
        ------
        ValueError
            If inversion is not defined for non-simulation modes, or if
            only one of ip_head/redis_password is specified.

        Notes
        -----
        Validation checks:
        - Inversion must be defined when mode is "inversion" or "mcmc_inversion"
        - Both ip_head and redis_password must be specified together (or neither)
        """
        # Check that inversion is defined for non-simulation modes
        if self.mode != "simulation" and self.inversion is None:
            raise ValueError(
                "Inversion must be defined when running outside simulation mode"
            )

        # Check ray multi-node parameters
        if (self.ip_head is None) != (self.redis_password is None):
            raise ValueError(
                "Both ip_head and redis_password must be specified together"
            )

        return self
