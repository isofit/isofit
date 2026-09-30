"""
Configuration loader utilities for pyconf.

This module provides functions to load and validate ISOFIT configuration files
from disk (YAML/JSON) or from Python dictionaries. It handles path expansion
relative to the configuration file directory and provides backward compatibility
with legacy configuration loading functions.
"""

import logging
import os
from pathlib import Path
from typing import Union

import yaml
from pydantic import ValidationError

from isofit.configs.config import Config
from isofit.core import common

Logger = logging.getLogger(__name__)


def create_new_config(config_file: Union[str, Path]) -> Config:
    """
    Load a config file from disk and create a validated Config object.

    This function provides backward compatibility with the legacy
    configs.create_new_config() function. It reads a YAML or JSON configuration
    file, expands all relative paths to be relative to the config file's directory,
    and validates the configuration using Pydantic.

    Parameters
    ----------
    config_file : str or Path
        Path to config file. Must be in JSON or YAML format. Can be absolute
        or relative to the current working directory.

    Returns
    -------
    Config
        Validated Config object with all paths expanded and all fields validated.

    Raises
    ------
    IOError
        If config file cannot be read or has invalid format (not JSON/YAML).
    ValidationError
        If config validation fails due to missing required fields, invalid
        values, or constraint violations.

    Examples
    --------
    >>> from isofit.configs import create_new_config
    >>> config = create_new_config("config.yaml")
    >>> config.get_config_errors()
    >>> print(config.input.measured_radiance_file)

    Notes
    -----
    All relative file paths in the configuration are expanded relative to the
    directory containing the config file, not the current working directory.
    This ensures consistent behavior regardless of where the script is run from.
    """
    config_file = str(config_file)
    Logger.info(f"Loading config file: {config_file}")

    try:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    except Exception as e:
        raise IOError(
            f"Unexpected configuration file type, only json and yaml supported:\n {e}"
        )

    # Expand paths relative to config file directory
    configdir, _ = os.path.split(os.path.abspath(config_file))
    config_dict = common.expand_all_paths(config_dict, configdir)

    # Validate and create config
    config = Config.model_validate(config_dict)

    return config


def load_config_dict(config_dict: dict, base_dir: str | None = None) -> Config:
    """
    Load a config from a dictionary.

    Creates a validated Config object from a Python dictionary, optionally
    expanding relative paths relative to a specified base directory.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with structure matching Config model schema.
        Should contain keys like 'input', 'output', 'forward_model', and
        'implementation'.
    base_dir : str, optional
        Base directory for expanding relative file paths in the configuration.
        If None, paths are not expanded and must be absolute. By default None.

    Returns
    -------
    Config
        Validated Config object.

    Raises
    ------
    ValidationError
        If config_dict does not match the expected schema or contains invalid values.

    Examples
    --------
    >>> config_dict = {
    ...     "input": {"measured_radiance_file": "data/radiance.mat"},
    ...     "output": {"dir": "output/"},
    ... }
    >>> config = load_config_dict(config_dict, base_dir="/path/to/project")
    >>> print(config.input.measured_radiance_file)
    PosixPath('/path/to/project/data/radiance.mat')

    Notes
    -----
    This function is useful for programmatically constructing configurations
    or for testing purposes. For loading from files, use create_new_config() instead.
    """
    if base_dir:
        config_dict = common.expand_all_paths(config_dict, base_dir)

    return Config.model_validate(config_dict)
