from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .engines import Engines


class Atmosphere(BaseModel):
    engine: Engines

    wavelength_file: Path | None = Field(
        default=None,
        description="Optional path to wavelength file for high-res atmospheric calculations",
        examples=[""],
    )

    lut_path: Path = Field(
        default=None,
        description="Path to the look up table directory",
    )

    sim_path: Path = Field(
        default="${output}/lut/sims",
        description="Path to the look up table directory",
    )

    lut_grid: dict[str, list[float]] = Field(default=None, description="")

    lut_subset: dict[str, Any] = Field(default=None, description="")
