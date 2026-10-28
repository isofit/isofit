"""
Kernel Flows engine configuration.

Kernel Flows is a machine learning-based RT emulator providing fast
atmospheric correction capabilities.
"""

from typing import Annotated, Literal, Optional

from pydantic import BeforeValidator, ConfigDict, Field, FilePath

from isofit.configs.base import SectionModel

from . import standardName


class KernelFlowsConfig(SectionModel):
    """
    Kernel Flows engine configuration.

    Kernel Flows is a machine learning emulator for atmospheric RT calculations,
    providing efficient atmospheric correction through learned representations.

    Examples
    --------
    The ``kernelflows`` base config leaves the emulator unset:

    >>> from isofit.configs.sections.atmosphere.engines import KernelFlowsConfig
    >>> kf = KernelFlowsConfig(base="kernelflows")
    >>> kf.name, kf.emulator_file
    ('kernelflows', None)

    Supplying the emulator and template files (they must exist):

    >>> kf = KernelFlowsConfig(
    ...     base="kernelflows",
    ...     emulator_file="kf_emulator.h5",
    ...     template_file="modtran_template.json",
    ... )

    Notes
    -----
    Kernel Flows is an experimental emulator under development. It may
    not be available in all ISOFIT installations.

    See Also
    --------
    sRTMnetConfig : Alternative ML-based RT emulator
    SixSConfig : Physics-based RT engine
    """

    section = "forward_model.atmosphere.engine"
    defaults = "kernelflows"

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["kernelflows"], BeforeValidator(standardName)] = Field(
        description=(
            'Engine identifier. Must be "kernelflows" or its alias "kf" '
            "(case-insensitive)"
        )
    )

    emulator_file: Optional[FilePath] = Field(
        description="Path to the trained Kernel Flows emulator model file",
    )

    template_file: Optional[FilePath] = Field(
        description=(
            "Path to the template configuration file for Kernel Flows parameters"
        ),
    )
