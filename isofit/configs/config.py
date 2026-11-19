import os

os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"

import logging

from pydantic import Field, ValidationError

Logger = logging.getLogger(__name__)

from isofit.configs.base import SectionModel
from isofit.configs.sections import io, pipeline
from isofit.configs.sections.atmosphere import AtmosphereConfig
from isofit.configs.sections.forward_model import ForwardModelConfig
from isofit.configs.sections.implementation import ImplementationConfig


class Config(SectionModel):
    """
    Main ISOFIT configuration model.

    This Pydantic model validates and manages all ISOFIT configuration parameters,
    including input/output files, forward model components (surface, atmosphere,
    instrument), and implementation settings.

    Examples
    --------
    With no arguments, the Config is built from the ``default`` base config:

    >>> from isofit.configs import Config
    >>> config = Config()
    >>> config.forward_model.atmosphere.engine.name
    'prebuilt'

    Start from an alternative base config with ``base``; the remaining keywords
    are deep-merged on top (see :class:`~isofit.configs.base.SectionModel`):

    >>> config = Config(
    ...     base="vswir",
    ...     forward_model={"surface": {"surface_file": "surface.mat"}},
    ...     implementation={"n_cores": 4},
    ... )
    >>> config.forward_model.atmosphere.engine.name
    'srtmnet'
    >>> config.get_config_errors()  # Validate inter-field constraints

    Notes
    -----
    This class uses Pydantic for automatic validation at construction time. Most
    validation errors will be raised when creating a Config instance. The
    get_config_errors() method performs additional inter-field validation that
    cannot be expressed as simple field constraints.
    """

    section = ""

    input: io.InputConfig = Field(
        description=(
            "Input file configuration specifying measured radiance, observation "
            "files, location files, and other input data sources"
        )
    )

    output: io.OutputConfig = Field(
        description=(
            "Output file configuration specifying where to write estimated states, "
            "reflectance, radiance, and other products"
        )
    )

    forward_model: ForwardModelConfig = Field(
        description=(
            "Forward model configuration containing the surface, atmosphere, and "
            "instrument models used to simulate measurements"
        ),
    )

    implementation: ImplementationConfig = Field(
        description=(
            "Implementation configuration specifying the operating mode (inversion, "
            "MCMC, simulation), Ray parallelization settings, and other runtime "
            "parameters"
        ),
    )

    # Retrieval pipeline sections, see isofit.retrieval
    log: pipeline.LogConfig = Field(
        description="Logging configuration for the retrieval pipeline",
    )

    # Stages, in pipeline order
    setup: pipeline.SetupStage = Field(description="Retrieval ``setup`` stage")
    preprocess: pipeline.PreprocessStage = Field(
        description="Retrieval ``preprocess`` stage"
    )
    scene: pipeline.SceneStage = Field(description="Retrieval ``scene`` stage")
    surface: pipeline.SurfaceStage = Field(description="Retrieval ``surface`` stage")
    segment: pipeline.SegmentStage = Field(description="Retrieval ``segment`` stage")
    presolve: pipeline.PresolveStage = Field(description="Retrieval ``presolve`` stage")
    background: pipeline.BackgroundStage = Field(
        description="Retrieval ``background`` stage"
    )
    configure: pipeline.ConfigureStage = Field(
        description="Retrieval ``configure`` stage"
    )
    lut: pipeline.LutConfig = Field(
        description=(
            "Retrieval ``lut`` stage, and the LUT grid heuristics of the "
            "retrieval pipeline"
        ),
    )
    invert: pipeline.InvertStage = Field(description="Retrieval ``invert`` stage")
    interpolate: pipeline.InterpolateStage = Field(
        description="Retrieval ``interpolate`` stage"
    )
    cleanup: pipeline.CleanupStage = Field(description="Retrieval ``cleanup`` stage")

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
        >>> from isofit.configs import Config
        >>> config = Config(
        ...     base="vswir",
        ...     forward_model={"surface": {"surface_file": "surface.mat"}},
        ... )
        >>> config.get_config_errors()  # No errors
        >>> config.implementation.mode = "simulation"  # Without a reflectance_file
        >>> config.get_config_errors()
        Traceback (most recent call last):
            ...
        AttributeError: Configuration error(s) found. See log for details.

        Notes
        -----
        The following inter-field validations are performed:
        - Simulation mode requires input.reflectance_file to be set
        - The surface checks of :meth:`SurfaceConfig.get_config_issues`, which
          read the surface model files
        """
        Logger.info("Checking config sections for configuration issues")

        errors = []
        warnings = []

        if self.forward_model.surface is not None:
            errors, warnings = self.forward_model.surface.get_config_issues()

        # Check inter-section validity
        # 1. Simulation mode requires reflectance_file
        if (
            self.implementation.mode == "simulation"
            and self.input.reflectance_file is None
        ):
            errors.append(
                "If implementation.mode is set to simulation, "
                "input.reflectance_file must be set"
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
