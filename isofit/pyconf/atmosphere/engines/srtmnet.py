from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

from isofit.data import env
from isofit.pyconf.validators import Exists

from . import standardName


class sRTMnet(BaseModel):
    name: Annotated[Literal["srtmnet"], BeforeValidator(standardName)]

    base_dir: Exists = Field(
        default=None,
        description="Base path to engine directory",
    )

    emulator_batch_size: int = Field(
        default=4096,
        ge=1,
        description="Batch size for sRTMnet predictions. Set smaller to reduce memory usage, larger for faster emulation.",
        examples=[512, 1024, 2048, 4096, 8192],
    )

    emulator_file: Exists = Field(
        default=None,
        description="",
    )

    emulator_aux_file: Exists = Field(
        default=None,
        description="",
    )

    @model_validator(mode="before")
    @classmethod
    def env_fallback(cls, data: dict, info: dict) -> dict:
        """
        If the base_dir is not defined, falls back to the env ini. If it is defined,
        updates the current env to the new base (does not save).

        Once the base is synchronized between config and ini, sets emulator_file and
        emulator_aux_file if not defined.
        """
        if base := data.get("base_dir"):
            env.changePath("srtmnet", base)
        data["base_dir"] = env.srtmnet

        if data.get("emulator_file") is None:
            data["emulator_file"] = env.path("srtmnet", key="srtmnet.file", rpe=False)

        if data.get("emulator_aux_file") is None:
            data["emulator_aux_file"] = env.path(
                "srtmnet", key="srtmnet.aux", rpe=False
            )

        return data

    @field_validator("emulator_file", mode="after")
    @classmethod
    def check_6c(cls, file: Path) -> Path:
        if file.suffix == ".6c":
            from isofit.atmosphere.engines.six_s import get_exe

            if "co2" not in get_exe(cls.base_dir, version=True):
                raise ValueError(
                    "sRTMnet 6C requires a CO2 version of 6S. Please use the isofit download CLI to pull a CO2 tag: https://github.com/isofit/6S/tags"
                )

        return file
