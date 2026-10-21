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
        description=(
            "Operating mode for ISOFIT. ``inversion`` performs standard retrieval, "
            "``mcmc_inversion`` uses Markov Chain Monte Carlo sampling for "
            "uncertainty quantification, ``simulation`` performs forward modeling "
            "only"
        )
    )

    inversion: Optional[InversionConfig] = Field(
        description=(
            "Inversion configuration. Required (with at least one window) for the "
            "``inversion`` and ``mcmc_inversion`` modes"
        )
    )

    n_cores: Optional[int] = Field(
        description=(
            "Number of CPU cores to use for parallel processing. If None, uses all "
            "available cores"
        )
    )

    task_inflation_factor: int = Field(
        description=(
            "Submit ``task_inflation_factor * n_cores`` tasks to Ray to improve load "
            "balancing. Higher values improve CPU utilization but increase overhead"
        )
    )

    ip_head: Optional[str] = Field(
        description=(
            "Ray parameter: IP address and port of the Ray head node for multi-node "
            'distributed runs (e.g. "192.168.1.1:6379"). Must be specified together '
            "with ``redis_password``. None runs on a single node"
        )
    )

    redis_password: Optional[str] = Field(
        description=(
            "Ray parameter: Redis password for multi-node runs. Must be specified "
            "together with ``ip_head``"
        )
    )

    ray_include_dashboard: bool = Field(
        description="Ray parameter: Enable the Ray dashboard for monitoring"
    )

    ray_temp_dir: str = Field(
        description=(
            "Ray temporary directory for the object store and logs. Useful on "
            "multiuser systems to isolate Ray instances"
        ),
    )

    ray_ignore_reinit_error: bool = Field(
        description=(
            "Tell Ray to ignore re-initialization errors. Convenient for running "
            "multiple ISOFIT instances sequentially"
        ),
    )

    io_buffer_size: int = Field(
        description=(
            "Size of chunks to read/process/write, in number of spectra. Larger "
            "values improve I/O efficiency but increase memory usage"
        ),
    )

    max_hash_table_size: int = Field(
        description=(
            "Maximum size of inversion hash tables for caching forward model results"
        )
    )

    per_pixel_heuristic_prior: bool = Field(
        description=(
            "Use a per-pixel heuristic prior (True) or an image-wide universal value "
            "(False). Per-pixel priors adapt to local scene characteristics"
        ),
    )

    debug_mode: bool = Field(
        description=(
            "Run in debug mode, which circumvents Ray parallelization for easier "
            "debugging"
        )
    )

    isofit_version: str = Field(
        default=__version__,
        description="ISOFIT version used. Automatically set to the current version",
    )

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
            If inversion or its windows are not defined for non-simulation
            modes, or if only one of ip_head/redis_password is specified.

        Notes
        -----
        Validation checks:
        - Inversion, with at least one window, must be defined when mode is
          "inversion" or "mcmc_inversion"
        - Both ip_head and redis_password must be specified together (or neither)
        """
        # Check that inversion is defined for non-simulation modes
        if self.mode != "simulation":
            if self.inversion is None:
                raise ValueError(
                    "Inversion must be defined when running outside simulation mode"
                )
            if not self.inversion.windows:
                raise ValueError(
                    "inversion.windows must be defined when running outside "
                    "simulation mode"
                )

        # Check ray multi-node parameters
        if (self.ip_head is None) != (self.redis_password is None):
            raise ValueError(
                "Both ip_head and redis_password must be specified together"
            )

        return self
