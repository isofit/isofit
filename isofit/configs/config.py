import os

os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"

import logging

from pydantic import BaseModel, Field, ValidationError

Logger = logging.getLogger(__name__)

from isofit.configs.sections import io
from isofit.configs.sections.atmosphere import Atmosphere
from isofit.configs.sections.forward_model import ForwardModel
from isofit.configs.sections.implementation import Implementation


class Config(BaseModel):
    """
    Main ISOFIT configuration model.

    This Pydantic model validates and manages all ISOFIT configuration parameters,
    including input/output files, forward model components (surface, atmosphere,
    instrument), and implementation settings.

    Attributes
    ----------
    input : io.Input
        Input file configuration specifying measured radiance, observation files,
        location files, and other input data sources.
    output : io.Output
        Output file configuration specifying where to write estimated states,
        reflectance, radiance, and other products.
    forward_model : ForwardModel
        Forward model configuration containing surface, atmosphere, and instrument
        models used to simulate measurements.
    implementation : Implementation
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

    input: io.Input = Field(description="Input file configuration")

    output: io.Output = Field(
        default_factory=io.Output, description="Output file configuration"
    )

    forward_model: ForwardModel = Field(
        default_factory=ForwardModel,
        description="Forward model configuration (surface, atmosphere, instrument)",
    )

    implementation: Implementation = Field(
        default_factory=Implementation,
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
        >>> config = Config(input=Input(), output=Output())
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
