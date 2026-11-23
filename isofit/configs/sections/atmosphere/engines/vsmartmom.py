"""
vSmartMOM engine configuration.

Vector simulated measurements of the atmosphere using radiative transfer based on the Matrix Operator Method.
"""

from typing import Annotated, Literal, Optional

from pydantic import BeforeValidator, ConfigDict, Field, FilePath, model_validator

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

    aot_ref: float = Field(
        default=0.550,
        description="Reference wavelength for aerosol optical depth (um)",
        examples=[0.550],
    )

    aerosol_mu: list[float] = Field(
        default=[0.1],
        min_length=1,
        max_length=2,
        description="Mean radii for fine and coarse aerosol size distributions (um). By default, only one mode is used",
        examples=[[0.1]],
    )

    aerosol_sigma: list[float] = Field(
        default=[1.75],
        min_length=1,
        max_length=2,
        description="Standard deviation of radii for fine and coarse aerosol size distributions (um). By default, only one mode is used",
        examples=[[1.75]],
    )

    aerosol_nr: list[float] = Field(
        default=[1.5],
        min_length=1,
        max_length=2,
        description="Real part of the aerosol refractive indices for fine and coarse aerosol size distributions. By default, only one mode is used",
        examples=[[1.5]],
    )

    aerosol_ni: list[float] = Field(
        default=[0.001],
        min_length=1,
        max_length=2,
        description="Imaginary part of the aerosol refractive indices for fine and coarse aerosol size distributions. By default, only one mode is used",
        examples=[[0.001]],
    )

    nstreams_mom: int = Field(
        default=8,
        description="Weighted streams per hemisphere for vSmartMOM radiative transfer. The advised minimum is 3 (Rayleigh contributes through m=2).",
        examples=[3, 8, 16],
    )

    write_sim: bool = Field(
        default=True,
        description="Leaving write_sim as True will write out the fine resolution csv(s) to disk, while False will hold in memory for resampling to instrument",
    )

    polarization_type: Literal[
        "Stokes_I()", "Stokes_IQ()", "Stokes_IQU()", "Stokes_IQUV()"
    ] = Field(
        default="Stokes_I()",
        description="Stokes vector components tracked: I (intensity), Q and U (linear polarization), and V (circular polarization)",
        examples=["Stokes_I()"],
    )

    architecture: Literal["CPU()", "default_architecture()", "GPU()"] = Field(
        default="CPU()",
        description="Compute architecture for vSmartMOM",
        examples=["default_architecture()"],
    )

    float_type: Literal["Float64", "Float32"] = Field(
        default="Float32",
        description="Floating point precision type for calculations (Float32, Float64)",
        examples=["Float64"],
    )

    wl_resolutions: list[tuple[float, float, float]] = Field(
        default=[
            (340.0, 1200.0, 20.0),
            (1200.0, 2510.0, 10.0),
        ],
        description="List of tuples defining spectral resolutions (start_wl_nm, end_wl_nm, wn_res_cm_1)",
        examples=[
            [
                (340.0, 490.0, 4.0),
                (490.0, 2510.0, 0.1),
            ]
        ],
    )

    template_file: Optional[FilePath] = Field(
        description="Template file used as the base configuration for vSmartMOM",
        examples=["vsmartmom_template.json"],
    )

    @model_validator(mode="after")
    def check_aerosols_match(self):
        """
        Validate lengths of each of the aerosol configs match (either of length 1 or 2).

        Returns
        -------
        VSmartMomConfig
            The validated config.

        Raises
        ------
        ValueError
            If lengths of aerosol configs do not match
        """

        counts = {
            "aerosol_mu": len(self.aerosol_mu),
            "aerosol_sigma": len(self.aerosol_sigma),
            "aerosol_nr": len(self.aerosol_nr),
            "aerosol_ni": len(self.aerosol_ni),
        }
        if len(set(counts.values())) > 1:
            msg = ", ".join(f"{name}={n}" for name, n in counts.items())
            raise ValueError(
                f"aerosol_mu, aerosol_sigma, aerosol_nr, and aerosol_ni must have the same length. Instead got: {msg}"
            )
        return self
