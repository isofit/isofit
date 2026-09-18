import os

os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"

import logging

from pydantic import BaseModel, Field, ValidationError

Logger = logging.getLogger(__name__)

from isofit.configs.sections import io
from isofit.configs.sections.atmosphere import AtmosphereConfig
from isofit.configs.sections.forward_model import ForwardModelConfig
from isofit.configs.sections.implementation import ImplementationConfig


class Config(BaseModel):
    """
    Main ISOFIT configuration model.

    This Pydantic model validates and manages all ISOFIT configuration parameters,
    including input/output files, forward model components (surface, atmosphere,
    instrument), and implementation settings.

    Attributes
    ----------
    input : io.InputConfig
        Input file configuration specifying measured radiance, observation files,
        location files, and other input data sources.
    output : io.OutputConfig
        Output file configuration specifying where to write estimated states,
        reflectance, radiance, and other products.
    forward_model : ForwardModelConfig
        Forward model configuration containing surface, atmosphere, and instrument
        models used to simulate measurements.
    implementation : ImplementationConfig
        Implementation configuration specifying operating mode (inversion, MCMC,
        simulation), Ray parallelization settings, and other runtime parameters.

    Examples
    --------
    >>> from isofit.configs import Config, load_config_dict
    >>> config_dict = {"input": {"measured_radiance_file": "data.mat"}}
    >>> config = load_config_dict(config_dict)
    >>> config.get_config_errors()  # Validate inter-field constraints

    Notes
    -----
    This class uses Pydantic for automatic validation at construction time. Most
    validation errors will be raised when creating a Config instance. The
    get_config_errors() method performs additional inter-field validation that
    cannot be expressed as simple field constraints.
    """

    input: io.InputConfig = Field(description="Input file configuration")

    output: io.OutputConfig = Field(
        default_factory=io.OutputConfig, description="Output file configuration"
    )

    forward_model: ForwardModelConfig = Field(
        default_factory=ForwardModelConfig,
        description="Forward model configuration (surface, atmosphere, instrument)",
    )

    implementation: ImplementationConfig = Field(
        default_factory=ImplementationConfig,
        description="Implementation configuration (mode, inversion, etc.)",
    )

    def get_config_errors(self) -> None:
        """
        Check configuration validity and raise errors if found.

        This method provides backward compatibility with the legacy
        Config.get_config_errors() method. With Pydantic, most validation
        happens at construction time, so this method primarily performs
        additional inter-field validation checks that span multiple sections.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If configuration errors are found during inter-field validation.
            All errors are logged before raising.

        Examples
        --------
        >>> config = Config(input=InputConfig(), output=OutputConfig())
        >>> config.get_config_errors()  # No errors
        >>> config.implementation.mode = "simulation"
        >>> config.input.reflectance_file = None
        >>> config.get_config_errors()  # Raises AttributeError

        Notes
        -----
        The following inter-field validations are performed:
        - Simulation mode requires input.reflectance_file to be set
        """
        Logger.info("Checking config sections for configuration issues")

        errors = []
        warnings = []

        # Check inter-section validity
        # 1. Simulation mode requires reflectance_file
        if (
            self.implementation.mode == "simulation"
            and self.input.reflectance_file is None
        ):
            errors.append(
                "If implementation.mode is set to simulation, input.reflectance_file must be set"
            )

        # Log warnings
        for w in warnings:
            Logger.warning(w)

        # Log and raise errors
        for e in errors:
            Logger.error(e)

        if len(errors) > 0:
            raise AttributeError("Configuration error(s) found. See log for details.")

        Logger.info("Configuration file checks complete, no errors found.")
