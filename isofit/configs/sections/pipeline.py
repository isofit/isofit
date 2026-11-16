"""
Input/output file configuration for ISOFIT.

This module defines Pydantic models for specifying input data files
(radiance, observation geometry, location, etc.) and output product files
(estimated reflectance, state vectors, uncertainties, etc.).
"""

import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator

from isofit.configs.base import SectionModel


class LogConfig(SectionModel):
    file: Optional[Path] = Field(description="...")

    level: Optional[str] = Field(description="...")

    append: Optional[bool] = Field(description="...")

    config: Optional[Path] = Field(description="...")

    @field_validator("level")
    @classmethod
    def check_level(cls, value):
        value = value.upper()
        levels = logging.getLevelNamesMapping()

        if value not in levels:
            raise ValueError(f"Must be one of {levels}")

        return value
