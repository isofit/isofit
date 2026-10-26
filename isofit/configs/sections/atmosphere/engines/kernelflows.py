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
    The emulator and template files must exist:

    >>> from isofit.configs import load_config_dict
    >>> config = load_config_dict({
    ...     "forward_model": {
    ...         "atmosphere": {
    ...             "engine": {
    ...                 "name": "kf",
    ...                 "emulator_file": "kf_emulator.h5",
    ...                 "template_file": "modtran_template.json",
    ...             }
    ...         }
    ...     }
    ... })
    >>> kf = config.forward_model.atmosphere.engine
    >>> kf.name
    'kernelflows'

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
