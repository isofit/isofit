"""
Kernel Flows engine configuration.

Kernel Flows is a machine learning-based RT emulator providing fast
atmospheric correction capabilities.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from isofit.configs.utils.validators import PathExists

from . import standardName


class KernelFlows(BaseModel):
    """
    Kernel Flows engine configuration.

    Kernel Flows is a machine learning emulator for atmospheric RT calculations,
    providing efficient atmospheric correction through learned representations.

    Attributes
    ----------
    name : Literal["kernelflows"]
        Engine identifier. Must be "kernelflows" or "kf" (alias).
    emulator_file : PathExists, optional
        Path to the trained Kernel Flows emulator model file. If None,
        uses default from environment.
    template_file : PathExists
        Path to template configuration file for Kernel Flows parameters.

    Examples
    --------
    >>> from isofit.configs.sections.atmosphere.engines import KernelFlows
    >>> kf = KernelFlows(
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
    sRTMnet : Alternative ML-based RT emulator
    SixS : Physics-based RT engine
    """

    name: Annotated[Literal["kernelflows"], BeforeValidator(standardName)]

    emulator_file: PathExists = Field(
        default=None,
        description="Path to the Kernel Flows emulator model file",
    )

    template_file: PathExists = Field(
        description="Path to Kernel Flows template configuration file",
    )
