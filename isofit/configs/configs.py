#! /usr/bin/env python3
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov

import logging
from collections import OrderedDict
from typing import Dict, List, Type
import os
from isofit.configs.sections.input_config import InputConfig
from isofit.configs.sections.output_config import OutputConfig
from isofit.configs.sections.forward_model_config import ForwardModelConfig
from isofit.configs.sections.implementation_config import ImplementationConfig
from isofit.configs.base_config import BaseConfigSection
from isofit.core import common
import yaml


class Config(BaseConfigSection):
    """
    Handles the reading and formatting of configuration files.  Please note - there are many ways to do this, some
    of which require fewer lines of code.  This method was chosen to facilitate more clarity when using / adding /
    modifying code, particularly given the highly flexible nature of Isofit.

    How to use:

    To add an additional parameter to an existing class, simply go to the relevant config (e.g. for forward_model go
    to sections/forward_model_config.py), and in the config class (e.g. ForwardModelConfig) add the parameter.  Also
    Add a hidden parameter with the _type suffix, which will be used to check that configs read the appropriate type.
    Add comments directly below, to be auto-appended to online documentation.
    Example::
        class GenericConfigSection(BaseConfigSection):
            _attribute_type = str
            attribute = 'my attribute'
            \"""str: attribute does whatever it happens to do\"""

    To validate that attributes have appropriate relationships or characteristics, use the hidden _check_config_validity
    method to add more detailed validation checks. Simply return a list of string descriptions of errors from the
    method as demonstrated::
        def _check_config_validity(self) -> List[str]:
            errors = list()
            if self.attribute_min >= self.attribute_max:
                errors.append('attribute_min must be less than attribute_max.')
            return errors

    """

    def __init__(self, configdict) -> None:

        self._input_type = InputConfig
        self.input = InputConfig({})
        """InputConfig: Input config. Holds all input file information.
        """

        self._output_type = OutputConfig
        self.output = OutputConfig({})
        """OutputConfig: Output config. Holds all output file information.
        """

        self._forward_model_type = ForwardModelConfig
        self.forward_model = ForwardModelConfig({})
        """ForwardModelConfig: forward_model config. Holds information about surface models, 
        radiative transfer models, and the instrument.
        """

        self._implementation_type = ImplementationConfig
        self.implementation = ImplementationConfig({})
        """ImplementationConfig: holds information regarding how isofit is to be run, including relevant sub-configs 
        (e.g. inversion information).
        """

        # Load sub-classes and attributes
        self.set_config_options(configdict)

    def get_config_as_dict(self) -> dict:
        """Get configuration options as a nested dictionary with delineated sections.

        Returns:
            Configuration options as a nested dictionary with delineated sections.
        """
        config = OrderedDict()
        for config_section in self._get_nontype_attributes():
            populated_section = getattr(self, config_section)
            config[config_section] = populated_section.get_config_options_as_dict()
        return config

    def get_config_errors(self):
        """
        Get configuration option errors by checking the validity of each config section.
        """
        logging.info("Checking config sections for configuration issues")

        errors = self.check_config_validity()

        for e in errors:
            logging.error(e)

        if len(errors) > 0:
            raise AttributeError('Configuration error(s) found.  See log for details.')

        logging.info('Configuration file checks complete, no errors found.')


def get_config_differences(config_a: Config, config_b: Config) -> Dict:
    differing_items = dict()
    dict_a = config_a.get_config_as_dict()
    dict_b = config_b.get_config_as_dict()
    all_sections = set(list(dict_a.keys()) + list(dict_b.keys()))
    for section in all_sections:
        section_a = dict_a.get(section, dict())
        section_b = dict_b.get(section, dict())
        all_options = set(list(section_a.keys()) + list(section_b.keys()))
        for option in all_options:
            value_a = section_a.get(option, None)
            value_b = section_b.get(option, None)
            if value_a != value_b:
                logging.debug(
                    "Configs have different values for option {} in section {}:  {} and {}".format(
                        option, section, value_a, value_b
                    )
                )
                differing_items.setdefault(section, dict())[option] = (value_a, value_b)
    return differing_items


def create_new_config(config_file: str) -> Config:
    """Load a config file from disk.
    Args:
        config_file: file to load config from.  Currently accepted formats: JSON and YAML

    Returns:
        Config object, having completed all necessary config checks
    """
    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
    except:
        raise IOError('Unexpected configuration file time, only json and yaml supported')

    configdir, f = os.path.split(os.path.abspath(config_file))

    config_dict = common.expand_all_paths(config_dict, configdir)
    config = Config(config_dict)

    return config
