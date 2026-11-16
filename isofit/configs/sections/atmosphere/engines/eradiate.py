"""
Eradiate engine configuration.

Eradiate is an open-source Monte Carlo RT code built on Mitsuba. This module
configures Eradiate for ISOFIT atmospheric correction.
"""

from typing import Annotated, Literal, Optional

from pydantic import BeforeValidator, ConfigDict, Field, FilePath

from isofit.configs.base import SectionModel

from . import standardName


class EradiateConfig(SectionModel):
    """
    Eradiate engine configuration.

    Eradiate is a Monte Carlo RT code for solar-reflected wavelengths. ISOFIT
    runs it in 6c mode, building LUT terms from three Lambertian albedo runs.

    Examples
    --------
    The ``eradiate`` base config uses the komodo absorption database and the
    continental aerosol, and leaves the geometry unset:

    >>> from isofit.configs.sections.atmosphere.engines import EradiateConfig
    >>> erd = EradiateConfig(base="eradiate")
    >>> erd.name, erd.eradiate_spp
    ('eradiate', 1000)

    Notes
    -----
    Eradiate is an optional dependency, install with
    ``pip install "isofit[eradiate]"``.

    Geometry is read from the solzen/solaz/viewzen/viewaz/elev/alt fields, and
    any not set fall back to the MODTRAN template. A LUT dimension of the same
    quantity always takes precedence over both.

    See Also
    --------
    SixSConfig : Alternative open-source RT engine
    LibRadTranConfig : Alternative open-source RT engine
    """

    section = "forward_model.atmosphere.engine"
    defaults = "eradiate"

    model_config = ConfigDict(extra="forbid")

    name: Annotated[Literal["eradiate"], BeforeValidator(standardName)] = Field(
        description='Engine identifier. Must be "eradiate" (case-insensitive)'
    )

    eradiate_absorption_database: Optional[str] = Field(
        description=(
            "Eradiate absorption database. Mono (gecko, komodo) or CKD (monotropa, "
            "mycena, panellus, tuber); the spectral mode is inferred from it"
        ),
        examples=["komodo", "panellus"],
    )

    eradiate_particle_properties: Optional[str] = Field(
        description="Eradiate aerosol particle-properties dataset",
        examples=["govaerts_2021-continental"],
    )

    eradiate_spp: Optional[int] = Field(
        ge=1,
        description="Monte Carlo samples per pixel",
        examples=[500, 1000],
    )

    template_file: Optional[FilePath] = Field(
        description=(
            "MODTRAN template file used for fallback geometry and gas/aerosol values"
        ),
    )

    elev: Optional[float] = Field(
        description="Surface elevation in kilometers (positive)",
    )

    alt: Optional[float] = Field(
        description=(
            "Sensor altitude in kilometers. None treats the sensor as being at TOA"
        ),
    )

    solzen: Optional[float] = Field(
        ge=0,
        le=180,
        description="Solar zenith angle in degrees (0-180)",
    )

    solaz: Optional[float] = Field(
        description="Solar azimuth angle in degrees",
    )

    viewzen: Optional[float] = Field(
        description="View zenith angle in degrees",
    )

    viewaz: Optional[float] = Field(
        description="View azimuth angle in degrees",
    )
