from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

from isofit.pyconf.validators import Exists

from . import standardName


class LibRadTran(BaseModel):
    name: Annotated[Literal["libradtran"], BeforeValidator(standardName)]

    base_dir: Exists = Field(
        default=None,
        description="Base path to engine directory",
    )

    reptran_band_model: Literal["coarse", "medium", "fine"] = Field(
        default="coarse",
        description="REPTRAN band model. Options: coarse (15cm-1), medium (5 cm-1), fine (1 cm-1)",
        examples=["coarse", "medium", "fine"],
    )

    kb_alpha_1: float = Field(
        description="King-Byrne Angstrom parameter (alpha 1). Setting to None uses default aerosol profile."
    )

    kb_alpha_2: float = Field(
        description="King-Byrne Angstrom parameter (alpha 2). Setting to None uses default aerosol profile."
    )

    gg_set: float = Field(
        description="Constant asymmetry parameter to represent aerosols"
    )

    gg_file: Path = Field(
        description="Path to an asymmetry depth file that overwrites default profile"
    )

    tau_file: Path = Field(
        description="Path to an optical depth file that overwrites default profile"
    )

    ssa_file: Path = Field(
        description="Path to a single scattering albedo file that overwrites default profile"
    )

    ssa_file: Path = Field(
        description="Path to a phase function moments file that overwrites default profile"
    )

    @model_validator(mode="before")
    @classmethod
    def env_fallback(cls, data: dict, info: dict) -> dict:
        """
        If the base_dir is not defined, falls back to the env ini. If it is defined,
        updates the current env to the new base (does not save).
        """
        if base := data.get("base_dir"):
            env.changePath("libradtran", base)
        data["base_dir"] = env.libradtran

        return data
