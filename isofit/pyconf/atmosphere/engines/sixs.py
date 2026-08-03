from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)

from isofit.pyconf.validators import Exists

from . import standardName


class SixS(BaseModel):
    name: Annotated[Literal["sixs"], BeforeValidator(standardName)]

    base_dir: Exists = Field(
        default=None,
        description="Base path to engine directory",
    )

    day: int = Field(ge=1, le=366, description="Day Parameter", examples=[1, 365])

    month: int = Field(ge=1, le=12, description="Month parameter", examples=[1, 12])

    elev: PositiveFloat = Field(
        description="Elevation parameter",
    )

    alt: PositiveFloat = Field(
        description="Altitude parameter",
    )

    solzen: float = Field(
        ge=0,
        le=180,
        description="Solar zenith parameter",
    )

    solaz: float = Field(
        description="Solar azimuth parameter",
    )

    viewzen: float = Field(
        description="View zenith parameter",
    )

    viewaz: float = Field(
        description="View azimuth parameter",
    )

    obs: Exists = Field(
        description="6S observation file",
    )

    earth_sun_distance_file: Exists = Field(
        description="Earth-Sun distance file",
    )

    irradiance_file: Exists = Field(
        description="Irradiance file",
    )

    @model_validator(mode="before")
    @classmethod
    def env_fallback(cls, data: dict, info: dict) -> dict:
        """
        If the base_dir is not defined, falls back to the env ini. If it is defined,
        updates the current env to the new base (does not save).
        """
        if base := data.get("base_dir"):
            env.changePath("sixs", base)
        data["base_dir"] = env.sixs

        return data
