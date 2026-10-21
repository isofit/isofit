"""
Kernel Flows engine configuration.

Kernel Flows is a machine learning-based RT emulator providing fast
atmospheric correction capabilities.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, FilePath

from . import standardName


class KernelFlowsConfig(BaseModel):
    """
    Kernel Flows engine configuration.

    Kernel Flows is a machine learning emulator for atmospheric RT calculations,
    providing efficient atmospheric correction through learned representations.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import KernelFlowsConfig
    >>> kf = KernelFlowsConfig(
    ...     name="kernelflows",
    ...     emulator_file="kf_emulator.pkl",
    ...     template_file="kf_template.json"
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
