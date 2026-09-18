"""
sRTMnet (surrogate Radiative Transfer Model neural network) engine configuration.

sRTMnet is a machine learning emulator trained on 6S simulations, providing
orders-of-magnitude speedup over physics-based RT codes while maintaining
high accuracy.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

from isofit.configs.utils.validators import PathExists
from isofit.data import env

from . import standardName


class sRTMnet(BaseModel):
    """
    sRTMnet engine configuration.

    sRTMnet is a neural network emulator trained on 6S atmospheric RT
    simulations. It provides near-instantaneous RT calculations suitable
    for real-time or large-scale processing.

    Attributes
    ----------
    name : Literal["srtmnet"]
        Engine identifier. Must be "srtmnet" or "kf" (alias).
    base_dir : PathExists
        Base path to sRTMnet installation directory. If not provided,
        falls back to environment configuration.
    emulator_batch_size : int
        Batch size for sRTMnet predictions. Larger batches are faster but
        require more GPU memory. Reduce if running out of memory.
        Default is 4096.
    emulator_file : PathExists
        Path to trained sRTMnet model file (.6s or other format). If not
        provided, uses default from environment.
    emulator_aux_file : PathExists
        Path to auxiliary emulator data file (normalization parameters, etc.).
        If not provided, uses default from environment.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import sRTMnet
    >>> srtm = sRTMnet(
    ...     name="srtmnet",
    ...     emulator_batch_size=2048,
    ...     emulator_file="srtmnet_model.6s",
    ...     emulator_aux_file="srtmnet_aux.npz"
    ... )

    Notes
    -----
    sRTMnet requires PyTorch and a GPU for optimal performance, though it
    can run on CPU. Training data and model weights are provided with ISOFIT.

    The .6c model variant includes CO2 absorption and requires a CO2-enabled
    version of 6S for training data consistency.

    sRTMnet is typically 1000x faster than 6S for LUT generation while
    maintaining <1% error in most conditions.

    See Also
    --------
    SixS : Physics-based RT code that sRTMnet emulates
    Prebuilt : Even faster option using pre-computed LUTs
    """

    name: Annotated[Literal["srtmnet"], BeforeValidator(standardName)]

    base_dir: PathExists = Field(
        default=None,
        description="Base path to engine directory",
    )

    emulator_batch_size: int = Field(
        default=4096,
        ge=1,
        description="Batch size for sRTMnet predictions. Set smaller to reduce memory usage, larger for faster emulation.",
        examples=[512, 1024, 2048, 4096, 8192],
    )

    emulator_file: PathExists = Field(
        default=None,
        description="",
    )

    emulator_aux_file: PathExists = Field(
        default=None,
        description="",
    )

    @model_validator(mode="before")
    @classmethod
    def env_fallback(cls, data: dict, info: dict) -> dict:
        """
        Handle base_dir and emulator file environment fallback.

        If base_dir is not defined in configuration, falls back to environment
        settings. If base_dir is defined, updates the current environment.
        After synchronizing base_dir, sets emulator_file and emulator_aux_file
        to defaults from environment if not explicitly specified.

        Parameters
        ----------
        data : dict
            Raw configuration data before validation.
        info : dict
            Validation context information from Pydantic.

        Returns
        -------
        dict
            Configuration data with paths set from config or environment defaults.

        Notes
        -----
        This allows sRTMnet installation and model files to be configured once
        in environment settings and reused across multiple configurations.
        """
        if base := data.get("base_dir"):
            env.changePath("srtmnet", base)
        data["base_dir"] = env.srtmnet

        if data.get("emulator_file") is None:
            data["emulator_file"] = env.path("srtmnet", key="srtmnet.file", rpe=False)

        if data.get("emulator_aux_file") is None:
            data["emulator_aux_file"] = env.path(
                "srtmnet", key="srtmnet.aux", rpe=False
            )

        return data

    @field_validator("emulator_file", mode="after")
    @classmethod
    def check_6c(cls, file: Path) -> Path:
        """
        Validate CO2 version compatibility for .6c models.

        The .6c model variant includes CO2 absorption and requires a
        CO2-enabled version of 6S for consistency.

        Parameters
        ----------
        file : Path
            Emulator file path to validate.

        Returns
        -------
        Path
            The validated file path.

        Raises
        ------
        ValueError
            If .6c model is used without CO2-enabled 6S installation.

        Notes
        -----
        CO2-enabled 6S versions are available from the isofit/6S repository
        with tags indicating CO2 support.
        """
        if file.suffix == ".6c":
            from isofit.atmosphere.engines.six_s import get_exe

            if "co2" not in get_exe(cls.base_dir, version=True):
                raise ValueError(
                    "sRTMnet 6C requires a CO2 version of 6S. Please use the isofit download CLI to pull a CO2 tag: https://github.com/isofit/6S/tags"
                )

        return file
