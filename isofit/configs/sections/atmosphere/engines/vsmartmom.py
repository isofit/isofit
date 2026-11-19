"""
vSmartMOM engine configuration.

Vector simulated measurements of the atmosphere using radiative transfer based on the Matrix Operator Method.
"""

from typing import Annotated, Literal, Optional

from pydantic import BeforeValidator, ConfigDict, Field, FilePath

from isofit.configs.base import SectionModel

from . import standardName


class VSmartMomConfig(SectionModel):
    """
    vSmartMOM engine configuration.

    Vector simulated measurements of the atmosphere using radiative transfer based on the Matrix Operator Method.
    """

    section = "forward_model.atmosphere.engine"
    defaults = "vsmartmom"

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["vsmartmom"], BeforeValidator(standardName)] = Field(
        description='Engine identifier. Must be ``"vsmartmom"``',
    )

    template_file: Optional[FilePath] = Field(
        description="Template file used as the base configuration for vSmartMOM",
        examples=["vsmartmom_template.json"],
    )

    aot_ref: float = Field(
        default=0.550,
        description="Reference wavelength for aerosol optical depth (um)",
        examples=[0.550],
    )

    aerosol_mu: float = Field(
        default=0.1,
        description="Log mean radius for aerosol size distribution (um)",
        examples=[0.1],
    )

    aerosol_sigma: float = Field(
        default=1.75,
        description="Log stddev of radius for aerosol size distribution (um)",
        examples=[1.75],
    )

    aerosol_nr: float = Field(
        default=1.5,
        description="Real part of the aerosol refractive index",
        examples=[1.5],
    )

    aerosol_ni: float = Field(
        default=0.001,
        description="Imaginary part of the aerosol refractive index",
        examples=[0.001],
    )

    nstreams_mom: int = Field(
        default=8,
        description="Weighted streams per hemisphere for vSmartMOM radiative transfer. The advised minimum is 3 (Rayleigh contributes through m=2).",
        examples=[3, 8],
    )

    architecture: Literal["CPU()", "default_architecture()", "GPU()", "MetalGPU()"] = (
        Field(
            default="CPU()",
            description="Compute architecture for vSmartMOM",
            examples=["default_architecture()"],
        )
    )

    float_type: str = Field(
        default="Float64",
        description="Floating point precision type for calculations (Float32, Float64)",
        examples=["Float64"],
    )

    wl_resolutions: list[tuple[float, float, float]] = Field(
        default=[
            (350.0, 1200.0, 20.0),
            (1200.0, 2510.0, 10.0),
        ],
        description="List of tuples defining spectral resolutions (start_wl_nm, end_wl_nm, wn_res_cm_1)",
        examples=[
            [
                (350.0, 490.0, 4.0),
                (490.0, 685.0, 2.0),
                (685.0, 960.0, 1.0),
                (960.0, 1345.0, 0.5),
                (1345.0, 1880.0, 0.3),
                (1880.0, 2510.0, 0.1),
            ]
        ],
    )
