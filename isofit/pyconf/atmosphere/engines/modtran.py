from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from isofit.pyconf.validators import Exists

from . import standardName


class Modtran(BaseModel):
    name: Annotated[Literal["modtran"], BeforeValidator(standardName)]

    base_dir: Exists = Field(
        default=None,
        description="Base path to engine directory",
    )

    multipart_transmittance: bool = Field(
        default=False,
        description="Apply triple-run diffuse & direct transmittance estimation",
        examples=[1, 365],
    )

    aerosol_template_file: Exists = Field(
        description="Aerosol template file",
    )

    aerosol_model_file: Exists = Field(
        description="Aerosol model file",
    )

    @model_validator(mode="before")
    @classmethod
    def env_fallback(cls, data: dict, info: dict) -> dict:
        """
        If the base_dir is not defined, falls back to the env ini. If it is defined,
        updates the current env to the new base (does not save).
        """
        if base := data.get("base_dir"):
            env.changePath("modtran", base)
        data["base_dir"] = env.modtran

        return data
