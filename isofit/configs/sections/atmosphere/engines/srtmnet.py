"""
sRTMnet (surrogate Radiative Transfer Model neural network) engine configuration.

sRTMnet is a machine learning emulator trained on 6S simulations, providing
orders-of-magnitude speedup over physics-based RT codes while maintaining
high accuracy.
"""

from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    model_validator,
)

from . import standardName


class sRTMnetConfig(BaseModel):
    """
    sRTMnet engine configuration.

    sRTMnet is a neural network emulator trained on 6S atmospheric RT
    simulations. It provides near-instantaneous RT calculations suitable
    for real-time or large-scale processing.

    Attributes
    ----------
    name : Literal["srtmnet"]
        Engine identifier. Must be "srtmnet" or "kf" (alias).
    base_dir : DirectoryPath
        Base path to sRTMnet installation directory. The base configs default
        this to the ISOFIT ini ``srtmnet`` path.
    sixs_dir : DirectoryPath
        Path to the 6S installation used to run the surrogate simulations.
        The base configs default this to the ISOFIT ini ``sixs`` path.
    emulator_batch_size : int
        Batch size for sRTMnet predictions. Larger batches are faster but
        require more GPU memory. Reduce if running out of memory.
        Default is 4096.
    emulator_file : FilePath
        Path to trained sRTMnet model file (.6s or other format). The base
        configs default this to the ISOFIT ini ``srtmnet.file``.
    emulator_aux_file : FilePath
        Path to auxiliary emulator data file (normalization parameters, etc.).
        The base configs default this to the ISOFIT ini ``srtmnet.aux``.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import sRTMnetConfig
    >>> srtm = sRTMnetConfig(
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
    SixSConfig : Physics-based RT code that sRTMnet emulates
    PrebuiltConfig : Even faster option using pre-computed LUTs
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["srtmnet"], BeforeValidator(standardName)] = Field()

    base_dir: Optional[DirectoryPath] = Field(
        description="Base path to engine directory",
    )

    sixs_dir: Optional[DirectoryPath] = Field(
        description="Path to the 6S installation used for the surrogate simulations",
    )

    emulator_batch_size: int = Field(
        ge=1,
        description="Batch size for sRTMnet predictions. Set smaller to reduce memory usage, larger for faster emulation.",
        examples=[512, 1024, 2048, 4096, 8192],
    )

    emulator_file: Optional[FilePath] = Field(
        description="",
    )

    emulator_aux_file: Optional[FilePath] = Field(
        description="",
    )

    template_file: Optional[FilePath] = Field(
        description="MODTRAN template file used to build the 6S surrogate simulations",
    )

    @model_validator(mode="after")
    def check_6c(self) -> "sRTMnetConfig":
        """
        Validate CO2 version compatibility for .6c models.

        The .6c model variant includes CO2 absorption and requires a
        CO2-enabled version of 6S for consistency.

        Returns
        -------
        sRTMnetConfig
            The validated config.

        Raises
        ------
        ValueError
            If .6c model is used without CO2-enabled 6S installation.

        Notes
        -----
        CO2-enabled 6S versions are available from the isofit/6S repository
        with tags indicating CO2 support.
        """
        if self.emulator_file and self.emulator_file.suffix == ".6c":
            from isofit.atmosphere.engines.six_s import get_exe

            if "co2" not in get_exe(self.sixs_dir, version=True):
                raise ValueError(
                    "sRTMnet 6C requires a CO2 version of 6S. Please use the isofit download CLI to pull a CO2 tag: https://github.com/isofit/6S/tags"
                )

        return self
