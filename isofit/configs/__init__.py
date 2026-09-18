"""
ISOFIT configuration system using Pydantic.

This package provides configuration loading and validation for ISOFIT using
Pydantic models. It supports loading from YAML/JSON files with automatic path
expansion, field validation, and backward compatibility with legacy formats.

The main entry points are:
- Config: Main configuration model
- create_new_config: Load and validate config from file
- load_config_dict: Create config from dictionary

Examples
--------
>>> from isofit.configs import create_new_config
>>> config = create_new_config("config.yaml")
>>> config.get_config_errors()  # Additional validation
>>> print(config.input.measured_radiance_file)
"""

from isofit.configs.config import Config
from isofit.configs.loader import create_new_config, load_config_dict

__all__ = ["Config", "create_new_config", "load_config_dict"]
